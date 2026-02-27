precision mediump float;

uniform vec4 uEnable;   // x=cheek, y=jaw, z=nose, w=forehead
uniform vec4 uOpacity;  // x=cheek, y=jaw, z=nose, w=forehead
uniform float uIntensity; // 0..1 global
uniform float uSigmaPx;   // softness in pixels (scaled by face size)
uniform vec2 uMaskSize;   // (w,h) pixels
uniform float uFaceWidthPx; // face width in mask pixels (for absolute tuning)
uniform sampler2D uFaceMaskTex; // face/skin clip mask (alpha in .r)
uniform vec2 uFaceMaskSize;     // (w,h) pixels of face mask texture
uniform float uErodePx;         // inward erosion radius in pixels (faceWidthPx * 0.01)
uniform mat3 uUvTransform;      // shared UV transform (identity unless overridden)

uniform vec2 uCheekPts[6];     // 3 per side (temple -> cheekbone -> medial cheek), NDC
uniform vec2 uJawPts[10];      // ear -> chin -> ear (downsampled), NDC
uniform vec2 uNoseStart[2];    // left/right, NDC
uniform vec2 uNoseEnd[2];      // left/right, NDC
uniform vec2 uForeheadPts[9];  // temple -> hairline -> temple (downsampled), NDC
uniform vec2 uFaceCenter;      // NDC
uniform vec2 uSideVis;         // (left,right) visibility weights from pose (0..1)

// Soft exclusions (avoid painting into eyes/lips).
uniform vec2 uEyeCenter[2]; // NDC
uniform vec2 uEyeAxis[2];   // unit axis (inner->outer), NDC basis
uniform vec2 uEyeRadii[2];  // (rx, ry), NDC
uniform vec2 uBrowCenter[2]; // NDC
uniform vec2 uBrowAxis[2];   // unit axis, NDC basis
uniform vec2 uBrowRadii[2];  // (rx, ry), NDC
uniform vec2 uLipCenter;    // NDC
uniform vec2 uLipRadii;     // (rx, ry), NDC
uniform vec2 uNoseTip;      // NDC (for tiny under-tip occlusion)
uniform vec2 uNostrilCenter[2]; // NDC
uniform vec2 uNostrilRadii[2];  // (rx, ry), NDC

varying vec2 vTexCoord;
varying vec2 vNdcPos;

vec2 ndcToPx(vec2 ndc) {
    return vec2(ndc.x * uMaskSize.x * 0.5, ndc.y * uMaskSize.y * 0.5);
}

float saturate(float x) { return clamp(x, 0.0, 1.0); }

float smoothMin(float a, float b, float k) {
    float kk = max(k, 1e-4);
    float h = clamp(0.5 + 0.5 * (b - a) / kk, 0.0, 1.0);
    return mix(b, a, h) - kk * h * (1.0 - h);
}

float stripFalloff(float distPx, float halfWidthPx, float blurPx, float minFloor) {
    float hw = max(halfWidthPx, 1e-3);
    float t = saturate(1.0 - distPx / hw);
    // Requirement curve: slower falloff for "lift".
    t = pow(t, 0.65);
    // Use blurPx as an extra edge-softening control (keeps gradient clean, avoids patches).
    float b = max(blurPx, 1e-3);
    float edge = saturate(1.0 - max(distPx - (hw - b), 0.0) / b);
    edge = smoothstep(0.0, 1.0, edge);
    t *= edge;
    // Core-only minimum floor (prevents disappearing), but never extend beyond the strip width.
    float inside = step(distPx, hw);
    return max(t, minFloor) * inside;
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

float distToSegmentPx2(vec2 pPx, vec2 aPx, vec2 bPx) {
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

// Returns (minDistPx, closestNdcX, closestNdcY) packed.
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

// Returns (minDistPx, closestNdcX, closestNdcY) packed.
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
    // Convert closest back into NDC (approx) for center-distance logic.
    vec2 cNdc = vec2(bestCpx.x / (uMaskSize.x * 0.5), bestCpx.y / (uMaskSize.y * 0.5));
    return vec3(bestD, cNdc.x, cNdc.y);
}

float gaussian(float dPx, float sigmaPx) {
    float s = max(sigmaPx, 0.5);
    float x = dPx / s;
    return exp(-0.5 * x * x);
}

// Returns (t, signedDistPx, absDistPx) where t is along a->b in [0,1],
// and signedDist is positive on the "up" side (toward higher NDC y).
vec3 segmentSignedInfoPx(vec2 pPx, vec2 aNdc, vec2 bNdc) {
    vec2 aPx = ndcToPx(aNdc);
    vec2 bPx = ndcToPx(bNdc);
    vec2 ab = bPx - aPx;
    float denom = max(dot(ab, ab), 1e-6);
    float t = clamp(dot(pPx - aPx, ab) / denom, 0.0, 1.0);
    vec2 cPx = aPx + ab * t;
    vec2 d = pPx - cPx;

    // Perp vector for "up/down" classification.
    vec2 perp = vec2(-ab.y, ab.x);
    float pLen = max(length(perp), 1e-6);
    perp /= pLen;
    if (perp.y < 0.0) perp = -perp; // ensure perp points upward (NDC+Y)

    float sd = dot(d, perp);
    return vec3(t, sd, abs(sd));
}

float erodedFaceMask(vec2 uv) {
    // True erosion is a min-filter over a disc. We approximate with a fixed set of taps
    // (enough to reliably stop blur/smear leaks at the silhouette).
    vec2 texel = vec2(1.0) / max(uFaceMaskSize, vec2(1.0));
    vec2 r = texel * max(uErodePx, 0.0);
    vec2 r2 = r * 0.5;

    float mn = texture2D(uFaceMaskTex, uv).r;

    // Primary ring (r).
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0,  r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0, -r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r.x,  r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r.x,  r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r.x, -r.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r.x, -r.y)).r);

    // Secondary ring (r/2) helps when r is small (<~2px).
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r2.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r2.x, 0.0)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0,  r2.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(0.0, -r2.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r2.x,  r2.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r2.x,  r2.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2( r2.x, -r2.y)).r);
    mn = min(mn, texture2D(uFaceMaskTex, uv + vec2(-r2.x, -r2.y)).r);

    return saturate(mn);
}

float orientedEllipseOutside(vec2 pNdc, vec2 cNdc, vec2 axis, vec2 radii) {
    vec2 d = pNdc - cNdc;
    vec2 ax = normalize(axis);
    vec2 ay = vec2(-ax.y, ax.x);
    float x = dot(d, ax) / max(radii.x, 1e-4);
    float y = dot(d, ay) / max(radii.y, 1e-4);
    float v = x * x + y * y; // 1 at boundary
    // 0 inside -> 1 outside with a soft rim.
    return smoothstep(0.70, 1.18, v);
}

void main() {
    vec2 pNdc = vNdcPos;
    vec2 pPx = ndcToPx(pNdc);

    vec2 uv = (uUvTransform * vec3(vTexCoord, 1.0)).xy;
    uv = clamp(uv, vec2(0.0), vec2(1.0));

    float sigma = uSigmaPx;
    // Packed RGBA mask output:
    // - R: contourCore (narrow, stronger)
    // - G: contourBlend (wide, softer)
    // - B: highlightCore (thin ridge)
    // - A: highlightBlend (soft halo)
    float contourCore = 0.0;
    float contourBlend = 0.0;
    float highlightCore = 0.0;
    float highlightBlend = 0.0;

    // ── Cheek contour: under cheekbone, ear -> mid cheek, fades toward center ──
    if (uEnable.x > 0.5) {
        // Requirement (FINAL):
        // - Cheekbone-driven placement: contour ONLY below the cheekbone curve
        //   (ear/temple → outer eye corner → mid-cheek).
        // - Strongest near ear, fades to zero before the nose.
        // - Clean gradients (no texture-driven shading).
        float coreBlur = uFaceWidthPx * 0.006;
        float blendBlur = uFaceWidthPx * 0.010;

        // Keep mask "energy" separated: core vs blend are packed into different channels.
        // Overall strengths are applied in the full-res relighting pass.

        // Left cheek: choose the closest segment of the cheekbone curve (0-1 or 1-2).
        vec3 infoL0 = segmentSignedInfoPx(pPx, uCheekPts[0], uCheekPts[1]);
        vec3 infoL1 = segmentSignedInfoPx(pPx, uCheekPts[1], uCheekPts[2]);
        float useL1 = step(infoL1.z, infoL0.z);
        float sdL = mix(infoL0.y, infoL1.y, useL1);  // signed distance to cheekbone curve (px)
        float tLseg = mix(infoL0.x, infoL1.x, useL1);
        float tL = mix(0.5 * tLseg, 0.5 + 0.5 * tLseg, useL1); // 0..1 along curve
        // Anatomical constraint: keep contour below the ridge, but DO NOT push it downward.
        // Use a soft sign gate centered at the ridge so strongest shading stays near the cheekbone.
        // segmentSignedInfoPx returns signedDist > 0 on the "up" side.
        // Softer ridge gate reduces patchiness and prevents hard "stripe" transitions.
        float belowSoftCore = max(uFaceWidthPx * 0.009, 1.10);
        float belowSoftBlend = max(uFaceWidthPx * 0.015, 1.50);
        float belowL_core = smoothstep(-belowSoftCore, belowSoftCore, -sdL);
        float belowL_blend = smoothstep(-belowSoftBlend, belowSoftBlend, -sdL);

        // Contour centerline is offset slightly BELOW the cheekbone curve.
        float contourOffsetPx = clamp(uFaceWidthPx * 0.0085, 1.2, 12.0);
        float dL = abs(sdL + contourOffsetPx);

        // Cheek visibility profile (TikTok/IG style):
        // - avoid a blob right at the ear
        // - peak on the cheek hollow (~1/3 forward)
        // - fade before mouth (end of segment)
        // Ear-strong profile: peak early, fade well before nose.
        float cheekProfileL =
            smoothstep(0.02, 0.10, tL) *
            (1.0 - smoothstep(0.38, 0.86, tL));
        cheekProfileL = max(cheekProfileL, 0.02);
        float taperL = smoothstep(0.00, 1.00, tL); // 0 at ear -> 1 toward outer eye
        float coreHalfWidthL = mix(uFaceWidthPx * 0.013, uFaceWidthPx * 0.009, taperL);
        float blendHalfWidthL = mix(uFaceWidthPx * 0.024, uFaceWidthPx * 0.016, taperL);

        float coreTL = stripFalloff(dL, coreHalfWidthL, coreBlur, 0.0) * cheekProfileL;
        float blendTL = stripFalloff(dL, blendHalfWidthL, blendBlur, 0.0) * cheekProfileL;
        coreTL *= belowL_core;
        blendTL *= belowL_blend;

        // Highlight (paired ridge) directly ABOVE the contour (under-eye → temple).
        // Place it at a fixed signed offset on the "up" side of the cheekbone line.
        // Stronger lift highlight: slightly higher offset above the contour ridge.
        float hlOffsetPx = clamp(uFaceWidthPx * 0.010, 1.3, 12.0);
        float dHlL = abs(sdL - hlOffsetPx);

        // Per requirements: highlight widths thinner than contour.
        float hlCoreHalfW = coreHalfWidthL * 0.55;
        float hlBlendHalfW = blendHalfWidthL * 0.50;

        // Keep highlight shorter than contour; avoid lower-cheek blobs.
        float hlLenL = smoothstep(0.10, 0.26, tL) * (1.0 - smoothstep(0.74, 0.96, tL));
        // Keep highlight on the "up" side and near the offset ridge.
        float aboveL = smoothstep(-belowSoftCore, belowSoftCore, sdL);
        float hlCoreL = stripFalloff(dHlL, hlCoreHalfW, coreBlur * 0.85, 0.0) * cheekProfileL * hlLenL * aboveL;
        float hlBlendL = stripFalloff(dHlL, hlBlendHalfW, blendBlur * 0.85, 0.0) * cheekProfileL * hlLenL * aboveL;

        // Explicitly avoid beard-growth / smile-line region near the mouth corner.
        // Approximate mouth corners from lip center + radii (no extra landmarks needed).
        vec2 mouthCornerL = uLipCenter + vec2(-uLipRadii.x * 0.98, uLipRadii.y * 0.10);
        vec2 mouthCornerR = uLipCenter + vec2( uLipRadii.x * 0.98, uLipRadii.y * 0.10);
        vec2 cornerRadii = vec2(uLipRadii.x * 0.70, uLipRadii.y * 0.78);
        vec2 dCL = (pNdc - mouthCornerL) / max(cornerRadii, vec2(1e-4));
        vec2 dCR = (pNdc - mouthCornerR) / max(cornerRadii, vec2(1e-4));
        float vCL = dot(dCL, dCL);
        float vCR = dot(dCR, dCR);
        float nearCornerL = smoothstep(1.55, 0.90, vCL);
        float nearCornerR = smoothstep(1.55, 0.90, vCR);
        float avoidMouthL = 1.0 - nearCornerL;

        // Right cheek: choose the closest segment of the cheekbone curve (3-4 or 4-5).
        vec3 infoR0 = segmentSignedInfoPx(pPx, uCheekPts[3], uCheekPts[4]);
        vec3 infoR1 = segmentSignedInfoPx(pPx, uCheekPts[4], uCheekPts[5]);
        float useR1 = step(infoR1.z, infoR0.z);
        float sdR = mix(infoR0.y, infoR1.y, useR1);
        float tRseg = mix(infoR0.x, infoR1.x, useR1);
        float tR = mix(0.5 * tRseg, 0.5 + 0.5 * tRseg, useR1);
        float belowSoftCoreR = belowSoftCore;
        float belowSoftBlendR = belowSoftBlend;
        float belowR_core = smoothstep(-belowSoftCoreR, belowSoftCoreR, -sdR);
        float belowR_blend = smoothstep(-belowSoftBlendR, belowSoftBlendR, -sdR);

        float dR = abs(sdR + contourOffsetPx);

        float cheekProfileR =
            smoothstep(0.02, 0.10, tR) *
            (1.0 - smoothstep(0.38, 0.86, tR));
        cheekProfileR = max(cheekProfileR, 0.02);
        float taperR = smoothstep(0.00, 1.00, tR);
        float coreHalfWidthR = mix(uFaceWidthPx * 0.013, uFaceWidthPx * 0.009, taperR);
        float blendHalfWidthR = mix(uFaceWidthPx * 0.024, uFaceWidthPx * 0.016, taperR);

        float coreTR = stripFalloff(dR, coreHalfWidthR, coreBlur, 0.0) * cheekProfileR;
        float blendTR = stripFalloff(dR, blendHalfWidthR, blendBlur, 0.0) * cheekProfileR;
        coreTR *= belowR_core;
        blendTR *= belowR_blend;

        float hlOffsetPxR = hlOffsetPx;
        float dHlR = abs(sdR - hlOffsetPxR);
        float hlLenR = smoothstep(0.10, 0.26, tR) * (1.0 - smoothstep(0.74, 0.96, tR));
        float aboveR = smoothstep(-belowSoftCoreR, belowSoftCoreR, sdR);
        float hlCoreHalfWR = coreHalfWidthR * 0.55;
        float hlBlendHalfWR = blendHalfWidthR * 0.50;
        float hlCoreR = stripFalloff(dHlR, hlCoreHalfWR, coreBlur * 0.85, 0.0) * cheekProfileR * hlLenR * aboveR;
        float hlBlendR = stripFalloff(dHlR, hlBlendHalfWR, blendBlur * 0.85, 0.0) * cheekProfileR * hlLenR * aboveR;

        float avoidMouthR = 1.0 - nearCornerR;

        coreTL *= avoidMouthL; blendTL *= avoidMouthL; hlCoreL *= avoidMouthL; hlBlendL *= avoidMouthL;
        coreTR *= avoidMouthR; blendTR *= avoidMouthR; hlCoreR *= avoidMouthR; hlBlendR *= avoidMouthR;

        // Pack: cheeks contribute to both core + blend; highlight ridge sits above/outward of contour.
        contourCore += uOpacity.x * (coreTL * uSideVis.x + coreTR * uSideVis.y);
        contourBlend += uOpacity.x * (blendTL * uSideVis.x + blendTR * uSideVis.y);
        highlightCore += uOpacity.x * (hlCoreL * uSideVis.x + hlCoreR * uSideVis.y);
        highlightBlend += uOpacity.x * (hlBlendL * uSideVis.x + hlBlendR * uSideVis.y);
    }

    // ── Jawline: subtle shadow along jaw edge, ear -> chin -> ear ──
    if (uEnable.y > 0.5) {
        // Requirement:
        // - thin shadow just UNDER the jaw edge
        // - fades DOWN into neck (not up onto the cheek)
        vec3 jawBest = closestOnJawPx(pPx);
        float d = jawBest.x;
        vec2 jawC = vec2(jawBest.y, jawBest.z);
        vec2 jawCPx = ndcToPx(jawC);
        float belowJaw = smoothstep(-max(uFaceWidthPx * 0.010, 1.0), max(uFaceWidthPx * 0.010, 1.0), (jawC.y - pNdc.y));
        float downPx = max(jawCPx.y - pPx.y, 0.0);
        float neckFade = gaussian(downPx, max(uFaceWidthPx * 0.035, 2.0));

        float halfW = max(uFaceWidthPx * 0.0085, 1.05);
        float blurPx = max(uFaceWidthPx * 0.010, 1.10);
        float m = stripFalloff(d, halfW, blurPx, 0.0) * belowJaw * neckFade;

        float side = smoothstep(0.10, 0.55, abs(pNdc.x - uFaceCenter.x));
        float lower = 1.0 - smoothstep(uFaceCenter.y - 0.58, uFaceCenter.y - 0.18, pNdc.y);
        float vis = (pNdc.x < uFaceCenter.x) ? uSideVis.x : uSideVis.y;
        contourBlend += uOpacity.y * m * mix(0.42, 0.86, side) * lower * vis;
    }

    // ── Nose: two soft vertical lines along the bridge (+ gentle tip shading) ──
    if (uEnable.z > 0.5) {
        // Requirement (FINAL):
        // - ONLY TWO lines along the bridge (NO center line, NO tip blob)
        // - Symmetrical
        // - Very soft, barely visible
        // - width = faceWidth * 0.01
        // - alpha = baseIntensity * 0.25  (baseIntensity is uOpacity.z)
        // - strongest at mid-bridge, fades toward top/bottom
        // - slight blur: blurRadius = faceWidth * 0.008
        float halfW = (uFaceWidthPx * 0.01) * 0.5;
        float blurPx = uFaceWidthPx * 0.008;
        float alpha = uOpacity.z * 0.25;

        float d0 = distToSegmentPx(pPx, uNoseStart[0], uNoseEnd[0]);
        float d1 = distToSegmentPx(pPx, uNoseStart[1], uNoseEnd[1]);
        float d = min(d0, d1);

        vec3 info0 = segmentSignedInfoPx(pPx, uNoseStart[0], uNoseEnd[0]);
        vec3 info1 = segmentSignedInfoPx(pPx, uNoseStart[1], uNoseEnd[1]);
        float t = clamp((info0.x + info1.x) * 0.5, 0.0, 1.0);

        // Mid-bridge peak with end fades (very soft).
        float mid = 1.0 - abs(t - 0.5) * 2.0;
        mid = smoothstep(0.0, 1.0, mid);
        mid = pow(mid, 1.25);
        float endFade = smoothstep(0.02, 0.20, t) * (1.0 - smoothstep(0.80, 0.98, t));
        float grad = mid * endFade;

        float m = stripFalloff(d, halfW, blurPx, 0.0) * grad;
        contourBlend += alpha * m;
    }

    // ── Nose bridge highlight: thin centered reflected light ─────────────────
    // Requirements:
    // - thin, centered stripe
    // - subtle, no shimmer
    // - strongest at mid-bridge, fades at ends
    {
        vec2 cA = (uNoseStart[0] + uNoseStart[1]) * 0.5;
        vec2 cB = (uNoseEnd[0] + uNoseEnd[1]) * 0.5;
        float dC = distToSegmentPx(pPx, cA, cB);

        vec3 infoC = segmentSignedInfoPx(pPx, cA, cB);
        float t = clamp(infoC.x, 0.0, 1.0);
        float endFade = smoothstep(0.06, 0.24, t) * (1.0 - smoothstep(0.76, 0.96, t));

        float halfW = max(uFaceWidthPx * 0.0045, 0.90);
        float blurPx = max(uFaceWidthPx * 0.010, 1.20);
        float core = stripFalloff(dC, halfW, blurPx, 0.0) * endFade;
        float halo = stripFalloff(dC, halfW * 1.90, blurPx * 1.10, 0.0) * endFade;
        highlightCore += core * 0.85;
        highlightBlend += halo * 0.75;
    }

    // ── Forehead/temples: soft shading along outer hairline, fades inward ──
    if (uEnable.w > 0.5) {
        vec3 best = closestOnForeheadPx(pPx);
        float d = best.x;
        vec2 cNdc = vec2(best.y, best.z);

        // Fade inward: compare radius from face center.
        vec2 cPx = ndcToPx(cNdc);
        vec2 fcPx = ndcToPx(uFaceCenter);
        float rHair = length(cPx - fcPx);
        float rHere = length(pPx - fcPx);
        float inward = saturate((rHair - rHere) / max(sigma * 7.0, 1.0)); // 1 near hairline → 0 deeper inside
        inward = pow(inward, 0.75);

        float temple = smoothstep(0.10, 0.55, abs(pNdc.x - uFaceCenter.x));
        float m = gaussian(d, sigma * 1.05) * inward * mix(0.65, 1.12, temple);
        float vis = (pNdc.x < uFaceCenter.x) ? uSideVis.x : uSideVis.y;
        contourBlend += uOpacity.w * m * vis;
    }

    // ── Forehead center highlight: soft reflected light patch ────────────────
    // Use the top-of-forehead point (uForeheadPts[4] corresponds to landmark 10).
    {
        vec2 top = uForeheadPts[4];
        vec2 c = mix(top, uFaceCenter, 0.24);
        // Wider horizontally, softer vertically.
        float rx = max(uFaceWidthPx * 0.022, 2.0) / (uMaskSize.x * 0.5);
        float ry = max(uFaceWidthPx * 0.014, 1.6) / (uMaskSize.y * 0.5);
        vec2 d = (pNdc - c) / max(vec2(rx, ry), vec2(1e-4));
        float v = dot(d, d);
        float core = smoothstep(1.0, 0.0, v) * smoothstep(0.0, 1.0, (1.25 - v));
        float halo = smoothstep(2.0, 0.0, v);
        // Keep this extremely subtle; final strength comes from relight pass.
        highlightCore += core * 0.55;
        highlightBlend += halo * 0.35;
    }

    // NOTE:
    // Hard constraint (Contour ≠ beard shadow): we do not generate any center-lower-face
    // shadows/highlights here (upper lip / philtrum, chin center, under-lip occlusion).
    // Those regions are excluded in the full-res composite pass via landmark masks.

    // ── Tiny under-tip nose occlusion shadow (very subtle) ───────────────────
    {
        vec2 tip = uNoseTip + vec2(0.0, -max(uFaceWidthPx * 0.004, 0.9) / (uMaskSize.y * 0.5));
        float rx = max(uFaceWidthPx * 0.0060, 0.9) / (uMaskSize.x * 0.5);
        float ry = max(uFaceWidthPx * 0.0040, 0.7) / (uMaskSize.y * 0.5);
        vec2 d = (pNdc - tip) / max(vec2(rx, ry), vec2(1e-4));
        float v = dot(d, d);
        float occ = smoothstep(1.9, 0.0, v);
        contourCore += occ * 0.20;
        contourBlend += occ * 0.12;
    }

    // Global intensity and clamp (keep in gradient range; no solid fills).
    contourCore = saturate(contourCore * uIntensity);
    contourBlend = saturate(contourBlend * uIntensity);
    highlightCore = saturate(highlightCore * uIntensity);
    highlightBlend = saturate(highlightBlend * uIntensity);

    // ── Exclusions: eyes + lips (never contour over them) ──
    float keep = 1.0;
    keep *= orientedEllipseOutside(pNdc, uEyeCenter[0], uEyeAxis[0], uEyeRadii[0]);
    keep *= orientedEllipseOutside(pNdc, uEyeCenter[1], uEyeAxis[1], uEyeRadii[1]);
    // Brows: keep temples/hairline shading from muddying brows.
    keep *= orientedEllipseOutside(pNdc, uBrowCenter[0], uBrowAxis[0], uBrowRadii[0]);
    keep *= orientedEllipseOutside(pNdc, uBrowCenter[1], uBrowAxis[1], uBrowRadii[1]);
    keep *= smoothstep(0.70, 1.20, dot((pNdc - uLipCenter) / max(uLipRadii, vec2(1e-4)),
                                       (pNdc - uLipCenter) / max(uLipRadii, vec2(1e-4))));
    // Nostrils: never allow nose contour to shade inside nostrils.
    keep *= smoothstep(0.70, 1.20, dot((pNdc - uNostrilCenter[0]) / max(uNostrilRadii[0], vec2(1e-4)),
                                       (pNdc - uNostrilCenter[0]) / max(uNostrilRadii[0], vec2(1e-4))));
    keep *= smoothstep(0.70, 1.20, dot((pNdc - uNostrilCenter[1]) / max(uNostrilRadii[1], vec2(1e-4)),
                                       (pNdc - uNostrilCenter[1]) / max(uNostrilRadii[1], vec2(1e-4))));
    contourCore *= keep;
    contourBlend *= keep;
    highlightCore *= keep;
    highlightBlend *= keep;

    // ── Face clip + clean edge fade (NO BLEED) ───────────────────────────────
    // 1) Strict eroded mask: hard stop for background bleed (keep)
    // 2) Extra fade based on distance-to-boundary estimate (prevents hard edges near silhouette)
    float faceEroded = erodedFaceMask(uv);
    float faceRaw = texture2D(uFaceMaskTex, uv).r;

    vec2 texel = vec2(1.0) / max(uFaceMaskSize, vec2(1.0));
    float ax = texture2D(uFaceMaskTex, uv + vec2(texel.x, 0.0)).r
             - texture2D(uFaceMaskTex, uv - vec2(texel.x, 0.0)).r;
    float ay = texture2D(uFaceMaskTex, uv + vec2(0.0, texel.y)).r
             - texture2D(uFaceMaskTex, uv - vec2(0.0, texel.y)).r;
    float grad = max(length(vec2(ax, ay)) * 0.5, 1e-4);
    // Approx distance in pixels to the face boundary (alpha~0.5 is the edge).
    float distToFaceBoundary = max((faceRaw - 0.5) / grad, 0.0);

    float edgeSoftnessPx = uFaceWidthPx * 0.01;
    float edgeFade = smoothstep(0.0, max(edgeSoftnessPx, 1.0), distToFaceBoundary);
    float clip = faceEroded * edgeFade;
    contourCore *= clip;
    contourBlend *= clip;
    highlightCore *= clip;
    highlightBlend *= clip;

    // Output:
    // - R: contourCore
    // - G: contourBlend
    // - B: highlightCore
    // - A: highlightBlend
    gl_FragColor = vec4(contourCore, contourBlend, highlightCore, highlightBlend);
}

