precision highp float;

// Per-region gradient mask output:
// R = cheek contour
// G = jaw slimming
// B = nose contour
// A = chin shadow (under-chin center)

uniform vec4 uEnable;   // x=cheek, y=jaw, z=nose, w=chin
uniform vec4 uStrength; // x=cheek, y=jaw, z=nose, w=chin (0..1)

uniform float uSigmaPx;     // blur proxy (face-scaled), used only for internal feather curves
uniform vec2 uMaskSize;     // (w,h) pixels
uniform float uFaceWidthPx; // face width in mask pixels
uniform float uLowAngleT;   // 0..1 (1 = camera below / looking up)

uniform sampler2D uFaceMaskTex; // face/skin clip mask (alpha in .r)
uniform vec2 uFaceMaskSize;     // (w,h) pixels of face mask texture
uniform float uErodePx;         // inward erosion radius in pixels
uniform mat3 uUvTransform;

// Primary anchors in "content NDC" (post-cropScale).
uniform vec2 uCheekPath[6];    // 3 per side: near ear/temple -> cheekbone -> medial cheek
uniform vec2 uJawPts[10];      // ear -> chin -> ear
uniform vec2 uNoseStart[2];    // left/right
uniform vec2 uNoseEnd[2];      // left/right
uniform vec2 uForeheadPts[9];  // unused (kept for compatibility)
uniform vec2 uFaceCenter;
uniform vec2 uSideVis;         // (left,right) visibility weights (0..1)

// Exclusions
uniform vec2 uEyeCenter[2];
uniform vec2 uEyeAxis[2];
uniform vec2 uEyeRadii[2];
uniform vec2 uLipCenter;
uniform vec2 uLipRadii;
uniform vec2 uNoseTip;
uniform vec2 uNostrilCenter[2];
uniform vec2 uNostrilRadii[2];

varying vec2 vTexCoord;
varying vec2 vNdcPos;

float saturate(float x) { return clamp(x, 0.0, 1.0); }

vec2 ndcToPx(vec2 ndc) {
    return vec2(ndc.x * uMaskSize.x * 0.5, ndc.y * uMaskSize.y * 0.5);
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

// Signed distance to segment line (px), positive on the "up" side (toward higher NDC y).
vec3 segmentSignedInfoPx(vec2 pPx, vec2 aNdc, vec2 bNdc) {
    vec2 aPx = ndcToPx(aNdc);
    vec2 bPx = ndcToPx(bNdc);
    vec2 ab = bPx - aPx;
    float denom = max(dot(ab, ab), 1e-6);
    float t = clamp(dot(pPx - aPx, ab) / denom, 0.0, 1.0);
    vec2 cPx = aPx + ab * t;
    vec2 d = pPx - cPx;
    vec2 perp = vec2(-ab.y, ab.x);
    float pLen = max(length(perp), 1e-6);
    perp /= pLen;
    if (perp.y < 0.0) perp = -perp;
    float sd = dot(d, perp);
    return vec3(t, sd, abs(sd));
}

float stripFalloff(float distPx, float halfWidthPx, float edgeSoftPx) {
    float hw = max(halfWidthPx, 1e-3);
    float t = saturate(1.0 - distPx / hw);
    // Slower falloff reads like a hollow rather than a painted stripe.
    t = pow(t, 0.70);
    float b = max(edgeSoftPx, 1e-3);
    float edge = saturate(1.0 - max(distPx - (hw - b), 0.0) / b);
    edge = smoothstep(0.0, 1.0, edge);
    return t * edge * step(distPx, hw);
}

float gaussian(float dPx, float sigmaPx) {
    float s = max(sigmaPx, 0.5);
    float x = dPx / s;
    return exp(-0.5 * x * x);
}

float erodedFaceMask(vec2 uv) {
    vec2 texel = vec2(1.0) / max(uFaceMaskSize, vec2(1.0));
    vec2 r = texel * max(uErodePx, 0.0);
    vec2 r2 = r * 0.5;
    float mn = texture2D(uFaceMaskTex, uv).r;
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0,  r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0, -r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r.x,  r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r.x,  r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r.x, -r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r.x, -r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r2.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r2.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0,  r2.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0, -r2.y)).r);
    return saturate(mn);
}

float orientedEllipseOutside(vec2 pNdc, vec2 cNdc, vec2 axis, vec2 radii) {
    vec2 d = pNdc - cNdc;
    vec2 ax = normalize(axis);
    vec2 ay = vec2(-ax.y, ax.x);
    float x = dot(d, ax) / max(radii.x, 1e-4);
    float y = dot(d, ay) / max(radii.y, 1e-4);
    float v = x * x + y * y;
    return smoothstep(0.70, 1.18, v); // 0 inside -> 1 outside
}

float distToJawPx(vec2 pPx) {
    float d = 1e9;
    for (int i = 0; i < 9; i++) {
        d = min(d, distToSegmentPx(pPx, uJawPts[i], uJawPts[i + 1]));
    }
    return d;
}

vec3 closestOnJawPx(vec2 pPx) {
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
    vec2 cNdc = vec2(bestCpx.x / (uMaskSize.x * 0.5), bestCpx.y / (uMaskSize.y * 0.5));
    return vec3(bestD, cNdc.x, cNdc.y);
}

vec3 closestOnForeheadPx(vec2 pPx) {
    float bestD = 1e9;
    vec2 bestCpx = vec2(0.0);
    for (int i = 0; i < 8; i++) {
        vec2 aPx = ndcToPx(uForeheadPts[i]);
        vec2 bPx = ndcToPx(uForeheadPts[i + 1]);
        vec2 ab = bPx - aPx;
        float denom = max(dot(ab, ab), 1e-6);
        float t = clamp(dot(pPx - aPx, ab) / denom, 0.0, 1.0);
        vec2 cPx = aPx + ab * t;
        float d = length(pPx - cPx);
        if (d < bestD) { bestD = d; bestCpx = cPx; }
    }
    vec2 cNdc = vec2(bestCpx.x / (uMaskSize.x * 0.5), bestCpx.y / (uMaskSize.y * 0.5));
    return vec3(bestD, cNdc.x, cNdc.y);
}

void main() {
    vec2 pNdc = vNdcPos;
    vec2 pPx = ndcToPx(pNdc);

    float cheek = 0.0;
    float jaw = 0.0;
    float nose = 0.0;
    float chin = 0.0;

    // Feather controls (heavily feathered like TikTok).
    float edgeSoft = max(uFaceWidthPx * 0.010, 1.2);
    float lowT = clamp(uLowAngleT, 0.0, 1.0);

    // ── CHEEK CONTOUR (most important): hollow below cheekbone; stop halfway to mouth ──
    if (uEnable.x > 0.5 && uStrength.x > 0.001) {
        // Low-angle placement: move contour higher (closer to cheekbone ridge).
        float contourOffsetPx = clamp(uFaceWidthPx * 0.014, 1.8, 18.0);
        contourOffsetPx *= (1.0 - 0.20 * lowT);
        // Choose closest segment for each cheek (0-1 or 1-2; 3-4 or 4-5).
        vec3 iL0 = segmentSignedInfoPx(pPx, uCheekPath[0], uCheekPath[1]);
        vec3 iL1 = segmentSignedInfoPx(pPx, uCheekPath[1], uCheekPath[2]);
        // Avoid a moving seam at the Voronoi boundary between segments by blending
        // smoothly when both segments are similarly close (common under motion/jitter).
        float seamPx = max(uFaceWidthPx * 0.006, 1.35);
        float wL1 = smoothstep(-seamPx, seamPx, (iL0.z - iL1.z)); // 0->seg0 closer, 1->seg1 closer
        float sdL = mix(iL0.y, iL1.y, wL1);
        float tLseg = mix(iL0.x, iL1.x, wL1);
        float tL = mix(0.5 * tLseg, 0.5 + 0.5 * tLseg, wL1);
        float belowGateL = smoothstep(-max(uFaceWidthPx * 0.018, 1.8), max(uFaceWidthPx * 0.018, 1.8), -sdL);
        float dL = abs(sdL + contourOffsetPx);

        vec3 iR0 = segmentSignedInfoPx(pPx, uCheekPath[3], uCheekPath[4]);
        vec3 iR1 = segmentSignedInfoPx(pPx, uCheekPath[4], uCheekPath[5]);
        float wR1 = smoothstep(-seamPx, seamPx, (iR0.z - iR1.z));
        float sdR = mix(iR0.y, iR1.y, wR1);
        float tRseg = mix(iR0.x, iR1.x, wR1);
        float tR = mix(0.5 * tRseg, 0.5 + 0.5 * tRseg, wR1);
        float belowGateR = smoothstep(-max(uFaceWidthPx * 0.018, 1.8), max(uFaceWidthPx * 0.018, 1.8), -sdR);
        float dR = abs(sdR + contourOffsetPx);

        // Visibility profile:
        // Start near ear, peak early, then fade out BEFORE mouth (critical).
        float profL = smoothstep(0.06, 0.18, tL) * (1.0 - smoothstep(0.42, 0.62, tL));
        float profR = smoothstep(0.06, 0.18, tR) * (1.0 - smoothstep(0.42, 0.62, tR));

        float taperL = smoothstep(0.0, 1.0, tL);
        float taperR = smoothstep(0.0, 1.0, tR);
        float halfWCoreL = mix(uFaceWidthPx * 0.020, uFaceWidthPx * 0.012, taperL);
        float halfWCoreR = mix(uFaceWidthPx * 0.020, uFaceWidthPx * 0.012, taperR);
        float halfWBlendL = halfWCoreL * 2.0;
        float halfWBlendR = halfWCoreR * 2.0;

        float mL = (stripFalloff(dL, halfWBlendL, edgeSoft) * 0.55 + stripFalloff(dL, halfWCoreL, edgeSoft) * 0.80) * profL * belowGateL;
        float mR = (stripFalloff(dR, halfWBlendR, edgeSoft) * 0.55 + stripFalloff(dR, halfWCoreR, edgeSoft) * 0.80) * profR * belowGateR;

        cheek = (mL * uSideVis.x + mR * uSideVis.y) * uStrength.x;
    }

    // ── JAW SLIMMING: thin shadow under jawline; stronger near ear ──
    if (uEnable.y > 0.5 && uStrength.y > 0.001) {
        vec3 jawBest = closestOnJawPx(pPx);
        float d = jawBest.x;
        vec2 jawC = vec2(jawBest.y, jawBest.z);
        vec2 jawCPx = ndcToPx(jawC);

        // Only below jaw edge (toward neck).
        float below = smoothstep(-max(uFaceWidthPx * 0.012, 1.4), max(uFaceWidthPx * 0.012, 1.4), (jawC.y - pNdc.y));
        float downPx = max(jawCPx.y - pPx.y, 0.0);
        float neckFade = gaussian(downPx, max(uFaceWidthPx * 0.040, 2.0));

        float halfW = max(uFaceWidthPx * 0.010, 1.15);
        float m = stripFalloff(d, halfW, edgeSoft) * below * neckFade;

        // Stronger near ear, softer near chin.
        float side = smoothstep(0.08, 0.55, abs(pNdc.x - uFaceCenter.x));
        float lower = 1.0 - smoothstep(uFaceCenter.y - 0.60, uFaceCenter.y - 0.18, pNdc.y);
        float vis = (pNdc.x < uFaceCenter.x) ? uSideVis.x : uSideVis.y;
        jaw = m * mix(0.55, 1.0, side) * lower * vis * uStrength.y;
    }

    // ── NOSE CONTOUR: two thin vertical soft shadows + tiny under-tip ──
    if (uEnable.z > 0.5 && uStrength.z > 0.001) {
        float halfW = max(uFaceWidthPx * 0.0050, 0.90);
        float blurPx = max(uFaceWidthPx * 0.0080, 1.0);

        float d0 = distToSegmentPx(pPx, uNoseStart[0], uNoseEnd[0]);
        float d1 = distToSegmentPx(pPx, uNoseStart[1], uNoseEnd[1]);
        float d = min(d0, d1);

        vec3 info0 = segmentSignedInfoPx(pPx, uNoseStart[0], uNoseEnd[0]);
        vec3 info1 = segmentSignedInfoPx(pPx, uNoseStart[1], uNoseEnd[1]);
        float t = clamp((info0.x + info1.x) * 0.5, 0.0, 1.0);
        float mid = 1.0 - abs(t - 0.5) * 2.0;
        mid = pow(smoothstep(0.0, 1.0, mid), 1.35);
        float endFade = smoothstep(0.06, 0.22, t) * (1.0 - smoothstep(0.78, 0.98, t));

        float line = stripFalloff(d, halfW, blurPx) * mid * endFade;

        // Small shadow under tip (very subtle).
        vec2 tip = uNoseTip + vec2(0.0, -max(uFaceWidthPx * 0.0038, 0.9) / (uMaskSize.y * 0.5));
        float rx = max(uFaceWidthPx * 0.0058, 0.9) / (uMaskSize.x * 0.5);
        float ry = max(uFaceWidthPx * 0.0042, 0.7) / (uMaskSize.y * 0.5);
        vec2 dd = (pNdc - tip) / max(vec2(rx, ry), vec2(1e-4));
        float v = dot(dd, dd);
        float underTip = smoothstep(1.75, 0.0, v) * 0.55;

        nose = (line + underTip) * uStrength.z;
    }

    // ── CHIN SHADOW (critical): center under-chin shadow (stronger for low angle) ──
    if (uEnable.w > 0.5 && uStrength.w > 0.001) {
        // Chin anchor is the mid jaw point (jaw polyline includes landmark 152 at index 5).
        vec2 chinP = uJawPts[5];
        float dropPx = max(uFaceWidthPx * 0.020, 2.0) * (1.0 + 0.65 * lowT);
        float dropNdc = dropPx / max(uMaskSize.y * 0.5, 1.0);
        vec2 c = chinP + vec2(0.0, -dropNdc);

        float rxPx = max(uFaceWidthPx * 0.045, 3.0) * (1.0 + 0.40 * lowT);
        float ryPx = max(uFaceWidthPx * 0.030, 2.4) * (1.0 + 0.55 * lowT);
        float rx = rxPx / max(uMaskSize.x * 0.5, 1.0);
        float ry = ryPx / max(uMaskSize.y * 0.5, 1.0);

        vec2 d = (pNdc - c) / max(vec2(rx, ry), vec2(1e-4));
        float v = dot(d, d);
        float core = smoothstep(1.85, 0.0, v);
        float halo = smoothstep(3.10, 0.0, v);

        // Keep it strictly under the jaw edge (avoid painting onto lips/cheeks).
        float under = smoothstep(uFaceCenter.y - 0.08, uFaceCenter.y - 0.42, pNdc.y);
        chin = (core * 0.85 + halo * 0.35) * under * uStrength.w;
    }

    // ── Exclusions: eyes, lips, nostrils ──
    float keep = 1.0;
    keep *= orientedEllipseOutside(pNdc, uEyeCenter[0], uEyeAxis[0], uEyeRadii[0]);
    keep *= orientedEllipseOutside(pNdc, uEyeCenter[1], uEyeAxis[1], uEyeRadii[1]);
    keep *= smoothstep(0.70, 1.20, dot((pNdc - uLipCenter) / max(uLipRadii, vec2(1e-4)),
                                       (pNdc - uLipCenter) / max(uLipRadii, vec2(1e-4))));
    keep *= smoothstep(0.70, 1.20, dot((pNdc - uNostrilCenter[0]) / max(uNostrilRadii[0], vec2(1e-4)),
                                       (pNdc - uNostrilCenter[0]) / max(uNostrilRadii[0], vec2(1e-4))));
    keep *= smoothstep(0.70, 1.20, dot((pNdc - uNostrilCenter[1]) / max(uNostrilRadii[1], vec2(1e-4)),
                                       (pNdc - uNostrilCenter[1]) / max(uNostrilRadii[1], vec2(1e-4))));

    cheek *= keep;
    jaw *= keep;
    nose *= keep;
    chin *= keep;

    // ── Face clip (stop background bleed) ──
    vec2 uv = (uUvTransform * vec3(vTexCoord, 1.0)).xy;
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    float clip = erodedFaceMask(uv);
    cheek *= clip;
    jaw *= clip;
    nose *= clip;
    chin *= clip;

    gl_FragColor = vec4(saturate(cheek), saturate(jaw), saturate(nose), saturate(chin));
}

