precision mediump float;

uniform sampler2D uFrameTex; // base frame (camera + foundation/concealer already applied)
uniform sampler2D uMaskTex;  // packed low-res mask (RGBA: contourCore, contourBlend, highlightCore, highlightBlend)
uniform vec2 uTexelSize;     // full-res texel size for micro-contrast taps

uniform float uBaseContour;   // user/style base (0..1)
uniform float uBaseHighlight; // user/style base (0..1)
uniform float uBaseMicro;     // user/style base (0..1)
uniform float uBaseSpec;      // user/style base (0..1, small)
uniform vec3 uContourColor;   // sRGB 0..1, skin-adapted cool-neutral taupe/brown

uniform float uFaceMeanY;  // 0..1 (EMA)
uniform float uFaceStdY;   // 0..1 (EMA)
uniform float uClipFrac;   // 0..1 (EMA)
uniform float uLightBias;  // 0..1 (EMA)
uniform float uCoolTone;   // 0..1 (0=natural shadow, 1=ashy/cool contour)

// Beard/stubble exclusion (geometry + texture cues).
//
// Geometry is passed in the same NDC space used by the mask generator so we can:
// - avoid darkening moustache + chin + jaw/beard regions
// - keep contour on correct bone-structure zones (cheek hollow / under-jaw only)
//
// Note: NDC here is post-cropScale (matches contour_*_vertex.glsl vNdcPos).
uniform vec2 uJawPts[10];   // NDC (ear -> chin -> ear)
uniform vec2 uLipCenter;    // NDC
uniform vec2 uLipRadii;     // NDC
uniform vec2 uMouthCorners[2]; // NDC (left, right)
uniform vec2 uFaceCenter;   // NDC
uniform float uFaceWidthNdc; // absolute scale proxy in NDC (|right-left|)

// Debug:
// 0 = normal render
// 1 = contour mask (after beard exclusion)
// 2 = highlight mask
// 3 = beard/stubble exclusion mask
uniform float uDebugMode;

varying vec2 vTexCoord;
varying vec2 vNdcPos;

float saturate(float x) { return clamp(x, 0.0, 1.0); }

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec2 ndcToPx(vec2 ndc) {
    // NDC -1..1 spans the full viewport: 2 NDC units == width px.
    return vec2(ndc.x * (0.5 / max(uTexelSize.x, 1e-6)),
                ndc.y * (0.5 / max(uTexelSize.y, 1e-6)));
}

float ellipseMask(vec2 p, vec2 c, vec2 r, float feather) {
    vec2 rr = max(r, vec2(1e-5));
    vec2 d = (p - c) / rr;
    float v = dot(d, d); // 1 at boundary
    float f = clamp(feather, 0.02, 1.0);
    // Inside -> 1, outside -> 0 with a soft rim.
    return 1.0 - smoothstep(1.0, 1.0 + 0.60 * f, v);
}

float distToSegmentPx(vec2 pPx, vec2 aNdc, vec2 bNdc) {
    vec2 aPx = ndcToPx(aNdc);
    vec2 bPx = ndcToPx(bNdc);
    vec2 ab = bPx - aPx;
    float denom = max(dot(ab, ab), 1e-6);
    float t = clamp(dot(pPx - aPx, ab) / denom, 0.0, 1.0);
    vec2 c = aPx + ab * t;
    return length(pPx - c);
}

float distToJawPx(vec2 pPx) {
    float d = 1e9;
    for (int i = 0; i < 9; i++) {
        d = min(d, distToSegmentPx(pPx, uJawPts[i], uJawPts[i + 1]));
    }
    return d;
}

// Returns closest point on the jaw polyline in NDC.
vec2 closestOnJawNdc(vec2 pPx) {
    float bestD = 1e9;
    vec2 bestCpx = vec2(0.0);
    for (int i = 0; i < 9; i++) {
        vec2 aPx = ndcToPx(uJawPts[i]);
        vec2 bPx = ndcToPx(uJawPts[i + 1]);
        vec2 ab = bPx - aPx;
        float denom = max(dot(ab, ab), 1e-6);
        float t = clamp(dot(pPx - aPx, ab) / denom, 0.0, 1.0);
        vec2 cPx = aPx + ab * t;
        float d = length(pPx - cPx);
        if (d < bestD) { bestD = d; bestCpx = cPx; }
    }
    // ndcToPx inverse: px = ndc * (0.5/texel) => ndc = px * (2*texel)
    return vec2(bestCpx.x * (2.0 * uTexelSize.x),
                bestCpx.y * (2.0 * uTexelSize.y));
}

// Signed distance (pixels) to the line through segment a->b.
// Positive means "above" (toward higher NDC Y).
float signedDistToLinePx(vec2 pPx, vec2 aNdc, vec2 bNdc) {
    vec2 aPx = ndcToPx(aNdc);
    vec2 bPx = ndcToPx(bNdc);
    vec2 ab = bPx - aPx;
    vec2 perp = normalize(vec2(-ab.y, ab.x));
    if (perp.y < 0.0) perp = -perp;
    return dot(pPx - aPx, perp);
}

// Small 9-tap bilateral estimate of the "base" luma. This acts like a guided/bilateral
// filter: we apply sculpting to the base, then add the original detail back so pores stay.
float bilateralBaseLuma(vec2 uv, vec3 centerRgb) {
    float yc = luma(centerRgb);
    // Range sigma: larger keeps base stable under sensor noise, small enough to preserve edges.
    float sigmaR = 0.075;
    float inv2R2 = 1.0 / max(2.0 * sigmaR * sigmaR, 1e-6);

    vec2 o = uTexelSize;
    // Spatial weights (approx Gaussian with sigma ~1.0px).
    float w0 = 0.40;
    float w1 = 0.12;
    float w2 = 0.06;

    float sum = yc * w0;
    float wsum = w0;

    // 4-neighbors (1px)
    vec3 c1 = texture2D(uFrameTex, uv + vec2( o.x, 0.0)).rgb;
    vec3 c2 = texture2D(uFrameTex, uv + vec2(-o.x, 0.0)).rgb;
    vec3 c3 = texture2D(uFrameTex, uv + vec2(0.0,  o.y)).rgb;
    vec3 c4 = texture2D(uFrameTex, uv + vec2(0.0, -o.y)).rgb;

    float y1 = luma(c1); float y2 = luma(c2); float y3 = luma(c3); float y4 = luma(c4);
    float r1 = exp(-(y1 - yc) * (y1 - yc) * inv2R2);
    float r2 = exp(-(y2 - yc) * (y2 - yc) * inv2R2);
    float r3 = exp(-(y3 - yc) * (y3 - yc) * inv2R2);
    float r4 = exp(-(y4 - yc) * (y4 - yc) * inv2R2);

    sum += y1 * (w1 * r1); wsum += (w1 * r1);
    sum += y2 * (w1 * r2); wsum += (w1 * r2);
    sum += y3 * (w1 * r3); wsum += (w1 * r3);
    sum += y4 * (w1 * r4); wsum += (w1 * r4);

    // 4 diagonals (1px, sqrt(2) distance)
    vec3 d1c = texture2D(uFrameTex, uv + vec2( o.x,  o.y)).rgb;
    vec3 d2c = texture2D(uFrameTex, uv + vec2(-o.x,  o.y)).rgb;
    vec3 d3c = texture2D(uFrameTex, uv + vec2( o.x, -o.y)).rgb;
    vec3 d4c = texture2D(uFrameTex, uv + vec2(-o.x, -o.y)).rgb;
    float yd1 = luma(d1c); float yd2 = luma(d2c); float yd3 = luma(d3c); float yd4 = luma(d4c);
    float rd1 = exp(-(yd1 - yc) * (yd1 - yc) * inv2R2);
    float rd2 = exp(-(yd2 - yc) * (yd2 - yc) * inv2R2);
    float rd3 = exp(-(yd3 - yc) * (yd3 - yc) * inv2R2);
    float rd4 = exp(-(yd4 - yc) * (yd4 - yc) * inv2R2);

    sum += yd1 * (w2 * rd1); wsum += (w2 * rd1);
    sum += yd2 * (w2 * rd2); wsum += (w2 * rd2);
    sum += yd3 * (w2 * rd3); wsum += (w2 * rd3);
    sum += yd4 * (w2 * rd4); wsum += (w2 * rd4);

    return sum / max(wsum, 1e-5);
}

void main() {
    vec3 rgb = texture2D(uFrameTex, vTexCoord).rgb;
    vec4 m = texture2D(uMaskTex, vTexCoord);

    float contourCore = m.r;
    float contourBlend = m.g;
    float highlightCore = m.b;
    float highlightBlend = m.a;

    float Y = luma(rgb);
    float Ybase = bilateralBaseLuma(vTexCoord, rgb);
    float detail = Y - Ybase;

    // Lighting-aware strength auto-tuning.
    //
    // Requirement:
    // - Flat lighting (low contrast) => BOOST contour + highlight slightly so it stays visible.
    // - Strong existing shadows (high contrast) => REDUCE to avoid doubling.
    float flat = 1.0 - smoothstep(0.10, 0.19, uFaceStdY); // 1 in flat light, 0 in shadowy/high-contrast
    float shadowy = smoothstep(0.16, 0.26, uFaceStdY);    // 0 in flat, 1 in strong shadows
    float exposureOk = smoothstep(0.12, 0.62, uFaceMeanY);
    float clipOk = (1.0 - uClipFrac);

    float contourStrength =
        uBaseContour *
        exposureOk *
        mix(1.22, 0.86, shadowy) *
        mix(1.00, 1.18, flat);

    float highlightStrength =
        uBaseHighlight *
        exposureOk *
        clipOk *
        mix(1.18, 0.88, shadowy) *
        mix(1.00, 1.18, flat);

    float microStrength =
        uBaseMicro *
        mix(0.90, 1.10, shadowy) *
        mix(1.18, 0.92, flat);

    float specStrength =
        uBaseSpec *
        (1.0 - uClipFrac) *
        mix(0.85, 1.15, uLightBias);

    // Reduce contour on the bright side, strengthen slightly on the shadow side.
    // uLightBias > 0.5 => left side brighter; uLightBias < 0.5 => right brighter.
    float tSide = (vTexCoord.x - 0.5) * 2.0;         // -1 left .. +1 right
    float biasX = (uLightBias - 0.5) * 2.0;          // -1 right-bright .. +1 left-bright
    float sideSign = tSide * biasX;                  // >0 shadow side, <0 bright side
    float wShadowSide = smoothstep(-0.12, 0.12, sideSign);
    contourStrength *= mix(0.78, 1.10, wShadowSide);
    highlightStrength *= mix(1.06, 0.92, wShadowSide);

    // Suppress contour in dark pixels (prevents dirty jaw/shadow buildup).
    float darkSuppress = smoothstep(0.15, 0.35, Y);

    // Combine core+blend masks (still keep blend around for region gating).
    float mContour = saturate(contourCore * 0.95 + contourBlend * 0.55) * darkSuppress;
    float mHighlight = saturate(highlightCore * 1.00 + highlightBlend * 0.60);

    // ── Beard/stubble exclusion (CRITICAL) ───────────────────────────────────
    // Requirements:
    // - use lower-face landmark region (mouth-to-jaw)
    // - use chroma/texture cues (darker + higher-frequency)
    // - subtract from contour so facial hair is not darkened
    //
    // We compute a soft exclusion mask in NDC, then modulate it using texture cues
    // in full-res space (more reliable than low-res mask space).
    vec2 pNdc = vNdcPos;
    vec2 pPx = ndcToPx(pNdc);

    // Face scale in pixels from NDC (stable across device sizes).
    float faceWidthPx = max(uFaceWidthNdc, 0.10) * (0.5 / max(uTexelSize.x, 1e-6));

    // Geometry-based likely beard zones.
    float moustache = ellipseMask(
        pNdc,
        uLipCenter + vec2(0.0, uLipRadii.y * 0.62),
        vec2(uLipRadii.x * 0.92, uLipRadii.y * 0.60),
        0.55
    );
    float chin = ellipseMask(
        pNdc,
        mix(uJawPts[5], uLipCenter, 0.44),
        vec2(max(uFaceWidthNdc * 0.18, 0.06), max(uFaceWidthNdc * 0.14, 0.05)),
        0.60
    );

    // Jaw band inside the jawline region (thin band, lower face only).
    float dJaw = distToJawPx(pPx);
    float jawNear = 1.0 - smoothstep(faceWidthPx * 0.030, faceWidthPx * 0.115, dJaw);
    float mouthY = uLipCenter.y;
    float lowerFace = 1.0 - smoothstep(mouthY - uFaceWidthNdc * 0.28, mouthY - uFaceWidthNdc * 0.05, pNdc.y);
    float jawRegion = jawNear * lowerFace;

    float beardGeo = saturate(max(moustache, max(chin, jawRegion)));

    // Texture cues (full-res):
    // - hair tends to be darker, low-chroma, and high-frequency vs skin
    float mx = max(rgb.r, max(rgb.g, rgb.b));
    float mn = min(rgb.r, min(rgb.g, rgb.b));
    float sat = mx - mn;
    // Simple local high-frequency estimate.
    vec3 rx1 = texture2D(uFrameTex, vTexCoord + vec2(uTexelSize.x, 0.0)).rgb;
    vec3 rx2 = texture2D(uFrameTex, vTexCoord - vec2(uTexelSize.x, 0.0)).rgb;
    vec3 ry1 = texture2D(uFrameTex, vTexCoord + vec2(0.0, uTexelSize.y)).rgb;
    vec3 ry2 = texture2D(uFrameTex, vTexCoord - vec2(0.0, uTexelSize.y)).rgb;
    float Yblur = (Y + luma(rx1) + luma(rx2) + luma(ry1) + luma(ry2)) * 0.2;
    float hf = abs(Y - Yblur);

    float hfGate = smoothstep(0.005, 0.022, hf);
    float darkGate = 1.0 - smoothstep(0.28, 0.48, Y);
    float satGate = 1.0 - smoothstep(0.14, 0.30, sat);
    float beardCue = saturate(hfGate * darkGate * satGate);

    // Lower-face exclusion band (hard constraint): region below mouth corners down to jawline
    // (inside face only; does not touch under-jaw/neck).
    vec2 jawC = closestOnJawNdc(pPx);
    float sdMouthPx = signedDistToLinePx(pPx, uMouthCorners[0], uMouthCorners[1]); // <0 below mouth
    float mouthFeatherPx = max(faceWidthPx * 0.010, 2.0);
    float belowMouth = smoothstep(mouthFeatherPx, -mouthFeatherPx, sdMouthPx);
    float jawInsetPx = max(faceWidthPx * 0.006, 1.5);
    float jawFeatherPx = max(faceWidthPx * 0.012, 2.5);
    float aboveJaw = smoothstep(-jawFeatherPx, jawFeatherPx, (pPx.y - ndcToPx(jawC).y) - jawInsetPx);
    float lowerFaceBand = saturate(belowMouth * aboveJaw);

    // Hard exclusions: never apply contour or highlight here.
    float hardExclude = saturate(max(moustache, max(chin, lowerFaceBand)));

    // Soft exclusion: strengthen where cues suggest actual stubble/beard texture.
    float beardMask = max(hardExclude, beardGeo * mix(0.25, 1.0, beardCue));

    // Apply exclusion.
    mContour *= (1.0 - saturate(beardMask));
    mHighlight *= (1.0 - saturate(hardExclude));

    // Texture-based separation from stubble: slightly reduce contour strength on hair-like pixels.
    contourStrength *= mix(1.0, 0.88, beardCue);

    // Local shadow suppression (adaptive strength):
    // If the pixel is already darker than its local neighborhood, don't pile more "makeup shadow" on top.
    float localShadow = saturate((Yblur - Y) / 0.060);

    // Midtone-focused shadow curve (avoid dirty shadows / blown highlights).
    float wContour = smoothstep(0.12, 0.45, Y) * (1.0 - smoothstep(0.60, 0.92, Y));
    // Highlight lift curve (mids/highs, minimal shadows).
    float wHighlight = smoothstep(0.20, 0.60, Y);

    // Reduce darkness (~10–15%) and adaptively suppress in already-shadowy zones.
    contourStrength *= 0.88 * mix(1.0, 0.60, localShadow);
    // In shadowy zones, keep highlight but slightly reduce it so we don't fight real lighting.
    highlightStrength *= mix(1.0, 0.85, localShadow);

    // Local contrast sculpt (subtle): brighten above + darken below without a heavy makeup look.
    float sculpt = (mHighlight * wHighlight) - (mContour * wContour);

    // Layered soft shaping:
    // - Contour behaves like a gentle multiply on luminance
    // - Highlight behaves like a gentle screen on luminance
    // Apply sculpt to the *base* layer (edge-aware), then add detail back.
    float Ynew = Ybase;
    float cAmt = saturate(contourStrength * wContour * mContour);
    float hAmt = saturate(highlightStrength * wHighlight * mHighlight);
    Ynew = Ynew * (1.0 - cAmt * 0.55);
    Ynew = 1.0 - (1.0 - Ynew) * (1.0 - hAmt * 0.70);
    Ynew += sculpt * (0.028 + 0.018 * flat) * exposureOk;

    // Re-evaluate weights at the updated luminance for smoother, more "filter-like" depth.
    float wContour2 = smoothstep(0.12, 0.45, Ynew) * (1.0 - smoothstep(0.60, 0.92, Ynew));
    float wHighlight2 = smoothstep(0.20, 0.62, Ynew);
    float cAmt2 = saturate(contourStrength * wContour2 * mContour);
    float hAmt2 = saturate(highlightStrength * wHighlight2 * mHighlight);
    Ynew = Ynew * (1.0 - cAmt2 * 0.38);
    Ynew = 1.0 - (1.0 - Ynew) * (1.0 - hAmt2 * 0.48);
    Ynew += sculpt * (0.016 + 0.012 * flat) * exposureOk;

    // Restore original detail (pores/skin texture), but slightly smooth inside contour so it reads
    // like makeup (clean gradients) rather than locking onto hair/noise texture.
    float detailScale = mix(1.0, 0.82, saturate(mContour * 1.10));
    Ynew += detail * detailScale;

    float d = clamp((Y - Yblur) * microStrength, -0.05, 0.05);
    float region = saturate(contourBlend * 0.8 + highlightBlend * 0.5);
    Ynew += d * region;

    // Subtle "sheen" only on highlight core + brighter pixels.
    float sheen = smoothstep(0.35, 0.85, Y) * highlightCore;
    Ynew += sheen * specStrength;

    Ynew = clamp(Ynew, 0.0, 1.0);

    // Reconstruct RGB in a luma/chroma space so we can:
    // - keep natural texture
    // - apply a cool-toned contour correction (ashy brown, less red/orange)
    // - keep highlight neutral (avoid muddy/painty look across skin tones)
    float Cb = (rgb.b - Y) * 0.564;
    float Cr = (rgb.r - Y) * 0.713;

    float contourMask = saturate(mContour * wContour);
    float highlightMask = saturate(mHighlight * wHighlight);

    // ── Debug output (mask verification) ─────────────────────────────────────
    if (uDebugMode > 0.5) {
        float v = 0.0;
        if (uDebugMode < 1.5) {
            v = saturate(mContour);
        } else if (uDebugMode < 2.5) {
            v = saturate(mHighlight);
        } else {
            v = saturate(hardExclude);
        }
        gl_FragColor = vec4(vec3(v), 1.0);
        return;
    }

    float cool = clamp(uCoolTone, 0.0, 1.0) * contourMask;
    // Reduce red/orange (Cr) and bias slightly cool/ashy (Cb).
    Cr *= (1.0 - 0.30 * cool);
    Cb *= (1.0 + 0.16 * cool);

    // Reduce saturation slightly in contoured regions (makeup read vs dirt/shadow).
    float desat = saturate(contourMask) * 0.12;
    Cr *= (1.0 - desat);
    Cb *= (1.0 - desat);

    // Skin-tone-adapted contour tint (physically-inspired shadow coloration, not "paint").
    float Yc = luma(uContourColor);
    float CbC = (uContourColor.b - Yc) * 0.564;
    float CrC = (uContourColor.r - Yc) * 0.713;
    float tintAmt = saturate(contourMask) * (0.22 + 0.22 * wContour2);
    Cb = mix(Cb, CbC, tintAmt);
    Cr = mix(Cr, CrC, tintAmt);

    // Keep highlight chroma slightly restrained (reads like light, not paint).
    float hlChroma = highlightMask * 0.55;
    Cr *= (1.0 - 0.10 * hlChroma);
    Cb *= (1.0 - 0.08 * hlChroma);

    // YCbCr -> RGB (approx BT.601).
    float R = Ynew + 1.403 * Cr;
    float G = Ynew - 0.344 * Cb - 0.714 * Cr;
    float B = Ynew + 1.773 * Cb;
    vec3 outRgb = clamp(vec3(R, G, B), 0.0, 1.0);

    gl_FragColor = vec4(outRgb, 1.0);
}

