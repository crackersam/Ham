precision mediump float;

uniform sampler2D uTexture;
uniform vec2  uTexelSize;
uniform float uSmooth;      // skin-smoothing intensity (renderer-level, 0–1)
uniform float uTone;        // warmth intensity, kept in sync with camera shader
uniform float uFoundAlpha;  // per-style foundation opacity (0–1)
uniform vec3  uFoundTint;   // foundation pigment tint (linear-ish RGB)
uniform float uFoundCoverage; // 0..1 coverage strength (tint blend)
uniform float uRadiusScale;    // blur radius scale (1.0 = default)
uniform float uConcealLift;    // 0..1 shadow lift for concealer regions
uniform float uConcealNeutralize; // 0..1 chroma neutralization for concealer regions
// Auto-correct: detect locally-dark patches and correct them (works even when foundation alpha is 0).
uniform float uAutoCorrect;      // 0..1 overall strength
uniform float uAutoThreshold;    // luma delta threshold (darker-than-local-average)
uniform float uAutoRadiusScale;  // wide local-average radius scale
uniform float uDebugFoundation; // 0 = normal, 1 = debug face mask
uniform vec3  uDebugColor;      // debug mask tint color

varying float vEdgeFactor;
varying vec2  vCamUV;

// Weighted cross blur — samples camera texture directly
vec3 smoothSkin(vec2 uv) {
    // Larger radius so the reference differs enough to be perceptible at phone resolutions,
    // while keeping the tap count low for compatibility/perf.
    // Back-compat: if uRadiusScale is left at 0, treat it as 1.0 (default look).
    float rs = uRadiusScale;
    if (rs < 0.001) rs = 1.0;
    vec2 t = uTexelSize * (4.2 * rs);
    vec3 c = texture2D(uTexture, uv).rgb;
    vec3 n = texture2D(uTexture, uv + vec2( 0.0,  t.y)).rgb;
    vec3 s = texture2D(uTexture, uv + vec2( 0.0, -t.y)).rgb;
    vec3 e = texture2D(uTexture, uv + vec2( t.x,  0.0)).rgb;
    vec3 w = texture2D(uTexture, uv + vec2(-t.x,  0.0)).rgb;
    return (c * 4.0 + n + s + e + w) / 8.0;
}

// Wide local average — used for dark-patch detection (same 5 taps, wider radius).
vec3 smoothSkinWide(vec2 uv) {
    float rs = uAutoRadiusScale;
    if (rs < 0.001) rs = 1.0;
    // Heavier sampling than the "beauty blur" reference:
    // - reduces the patch's self-influence (better neighborhood estimate)
    // - more stable under noise/motion (under-eye cast varies frame-to-frame)
    vec2 t = uTexelSize * (10.5 * rs);

    vec3 c  = texture2D(uTexture, uv).rgb;
    vec3 n1 = texture2D(uTexture, uv + vec2( 0.0,  t.y)).rgb;
    vec3 s1 = texture2D(uTexture, uv + vec2( 0.0, -t.y)).rgb;
    vec3 e1 = texture2D(uTexture, uv + vec2( t.x,  0.0)).rgb;
    vec3 w1 = texture2D(uTexture, uv + vec2(-t.x,  0.0)).rgb;

    vec3 ne = texture2D(uTexture, uv + vec2( t.x,  t.y)).rgb;
    vec3 nw = texture2D(uTexture, uv + vec2(-t.x,  t.y)).rgb;
    vec3 se = texture2D(uTexture, uv + vec2( t.x, -t.y)).rgb;
    vec3 sw = texture2D(uTexture, uv + vec2(-t.x, -t.y)).rgb;

    // Extra far taps to better represent the surrounding skin tone when the patch
    // occupies much of the local area (e.g., eye socket shadow).
    vec2 tf = t * 2.05;
    vec3 n2 = texture2D(uTexture, uv + vec2( 0.0,  tf.y)).rgb;
    vec3 s2 = texture2D(uTexture, uv + vec2( 0.0, -tf.y)).rgb;
    vec3 e2 = texture2D(uTexture, uv + vec2( tf.x,  0.0)).rgb;
    vec3 w2 = texture2D(uTexture, uv + vec2(-tf.x,  0.0)).rgb;

    // Weighted sum: center dominates slightly; diagonals + far taps reduce bias.
    vec3 sum =
        c  * 4.0 +
        (n1 + s1 + e1 + w1) * 1.0 +
        (ne + nw + se + sw) * 0.75 +
        (n2 + s2 + e2 + w2) * 0.60;
    float norm = 4.0 + 4.0 * 1.0 + 4.0 * 0.75 + 4.0 * 0.60;
    return sum / norm;
}

// Lightweight YCbCr-ish helpers (no gamma correction; good enough for subtle tone evening).
vec3 rgbToYcc(vec3 rgb) {
    float y  = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    float cb = (rgb.b - y) * 0.564 + 0.5;
    float cr = (rgb.r - y) * 0.713 + 0.5;
    return vec3(y, cb, cr);
}

vec3 yccToRgb(vec3 ycc) {
    float y = ycc.x;
    float cb = ycc.y - 0.5;
    float cr = ycc.z - 0.5;
    float r = y + 1.403 * cr;
    float b = y + 1.773 * cb;
    float g = y - 0.344 * cb - 0.714 * cr;
    return vec3(r, g, b);
}

void main() {
    // Two-stage feather: same curve as makeup_fragment so all layers blend
    // consistently — sharper near-centre ramp, longer gentle outer fade
    float edgeAlpha = smoothstep(0.0, 0.20, vEdgeFactor)
                    * smoothstep(0.0, 0.55, vEdgeFactor);

    vec3 baseColor = texture2D(uTexture, vCamUV).rgb;
    vec3 color = baseColor;
    float luma  = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Foundation strength driver:
    // - uFoundAlpha controls opacity, but we also use it to scale the *correction* so
    //   the foundation reads visibly stronger instead of just "blending the same image".
    // Keep this closer to uFoundAlpha so styles remain meaningfully distinct.
    float foundStrength = clamp(uFoundAlpha * 1.45, 0.0, 1.0);

    // Always compute a blur reference so tone-evening has something to push toward,
    // even when uSmooth is low.
    vec3 blurred = smoothSkin(vCamUV);

    // Wide local reference for dark-patch detection. Only sampled when needed
    // (auto-correct, concealer, or patch debug).
    float autoStrength = clamp(uAutoCorrect, 0.0, 1.0);
    float concealLift0 = clamp(uConcealLift, 0.0, 1.0);
    float concealNeut0 = clamp(uConcealNeutralize, 0.0, 1.0);
    float needWide = step(0.001, autoStrength + concealLift0 + concealNeut0)
                   + step(2.5, uDebugFoundation); // patch debug uses mode 3+
    vec3 wide = blurred;
    if (needWide > 0.5) {
        wide = smoothSkinWide(vCamUV);
    }

    // Skin likelihood gate (midtone bias; keeps non-skin + highlights less affected).
    // Loosened slightly so deep under-eye shadows (low luma) still qualify as skin.
    float skinLikely = smoothstep(0.08, 0.38, luma)
                     * (1.0 - smoothstep(0.55, 0.95, luma));

    // Local-detail mask: if blur diverges from original, it's likely an edge/texture.
    // Reduce correction there to keep features crisp (brows/lashes).
    float detail = length(color - blurred);               // ~0..0.3 typical
    // Relaxed gate: preserve true edges, but allow more correction for mid-frequency
    // blemishes so spots/dents reduce without turning the whole face into blur.
    float keepDetail = 1.0 - smoothstep(0.045, 0.17, detail);

    // ── Auto dark-patch correction (face oval) ───────────────────────────────
    // Detect pixels that are darker than their immediate neighborhood and correct
    // only those regions. This runs even when foundation alpha is 0, but remains
    // detail-preserving and highlight-safe.
    float patchMask = 0.0;
    if (needWide > 0.5) {
        vec3 yccNow = rgbToYcc(baseColor);
        vec3 yccRef = rgbToYcc(wide);
        float lumaNow = yccNow.x;
        float lumaRef = yccRef.x;

        // Luma-based trigger: darker-than-local-average.
        float delta = max(0.0, lumaRef - lumaNow);
        float th = max(uAutoThreshold, 0.0);
        float knee = 0.060; // activation smoothness; keeps it stable under noise
        float rawLuma = smoothstep(th, th + knee, delta);

        // Chroma-cast trigger: detect brown/purple/blue-ish cast even when luma delta is small.
        // Weight Cr slightly higher; under-eye shadows often show up more in the red axis.
        vec2 dcc = yccNow.yz - yccRef.yz;
        float castDelta = length(dcc * vec2(1.0, 1.28));

        // Adaptive cast threshold: in deep shadows we require a stronger cast signal
        // to avoid chasing sensor noise; in midtones we allow gentler activation.
        float shadow = smoothstep(0.00, 0.28, 0.28 - lumaNow); // 0 mid/high → 1 deep shadow
        float castTh0 = 0.022 + 0.012 * shadow;
        float castKnee = 0.055;
        float rawCast = smoothstep(castTh0, castTh0 + castKnee, castDelta);

        // Apply where it reads as skin, and avoid lifting highlights.
        float highlightSafe = 1.0 - smoothstep(0.62, 0.92, lumaNow);
        float texKeep = 0.55 + 0.45 * keepDetail;

        // Chroma band-pass for "skin-likely": excludes near-neutral (background/white objects)
        // and very high chroma (lips/eyeshadow), while allowing typical skin chroma.
        float chromaMag = length(yccNow.yz - vec2(0.5));
        float skinChromaLikely =
            smoothstep(0.010, 0.065, chromaMag) *
            (1.0 - smoothstep(0.090, 0.220, chromaMag));

        float skinGate = skinLikely * skinChromaLikely;
        float gate = highlightSafe * (0.50 + 0.50 * skinGate) * texKeep;

        // Split masks so we can neutralize cast without necessarily lifting luma.
        float patchLuma = rawLuma * gate;
        float patchCast = rawCast * gate;
        patchMask = max(patchLuma, patchCast);

        if (autoStrength > 0.001) {
            vec3 ycc = rgbToYcc(color);

            // Neutralize cast primarily where cast differs, not just where it's dark.
            float neutAmt = clamp(autoStrength * patchCast * 1.45, 0.0, 1.0);
            ycc.yz = mix(ycc.yz, yccRef.yz, neutAmt);
            ycc.yz = mix(ycc.yz, vec2(0.5), neutAmt * 0.36);

            // Lift luminance primarily where the pixel is locally darker.
            float liftAmt = clamp(autoStrength * patchLuma * 1.35, 0.0, 1.0);
            // Slight “concealer-like” overshoot above the neighborhood average when we're confident
            // it's a true locally-dark patch. Kept small and highlight-safe by patchLuma gating.
            float targetY = max(ycc.x, clamp(yccRef.x + patchLuma * 0.012, 0.0, 1.0));
            ycc.x = mix(ycc.x, targetY, liftAmt);

            color = yccToRgb(ycc);
            luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        }
    }

    // Baseline "coverage": stronger low-frequency mix so foundation is visible
    // even in indoor lighting / motion blur, while still preserving sharp features.
    float coverage = foundStrength * keepDetail * (0.58 + 0.22 * skinLikely);
    color = mix(color, blurred, clamp(coverage, 0.0, 1.0));
    luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Luminance-gated, detail-preserving skin smoothing.
    if (uSmooth > 0.01) {
        float smoothAmt = uSmooth
                        * (0.55 + 0.85 * foundStrength)
                        * (0.35 + 0.65 * skinLikely)
                        * keepDetail;
        color = mix(color, blurred, clamp(smoothAmt, 0.0, 1.0) * 0.98);
        luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    }

    // Subtle warmth + saturation boost — kept identical to camera_fragment.glsl
    // so the foundation surface blends seamlessly with the unfiltered background
    if (uTone > 0.01) {
        color = mix(vec3(luma), color, 1.0 + uTone * 0.12);
        color += vec3(0.018, 0.010, 0.004) * uTone
                 * smoothstep(0.3, 0.65, luma);
        luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    }

    // ── Foundation finish: tone evening (stronger than blur, still natural) ──
    // Blend chroma toward the blurred sample while preserving luminance detail.
    // Scaled by uFoundAlpha so styles control the perceived strength.
    float chromaKeep = 1.0 - smoothstep(0.075, 0.28, detail);
    float chromaAmt = foundStrength * (0.35 + 0.65 * skinLikely) * chromaKeep;
    if (chromaAmt > 0.001) {
        vec3 ycc = rgbToYcc(color);
        vec3 yccB = rgbToYcc(blurred);

        // Coverage / tone evening: smooth chroma more than luma (keeps texture).
        ycc.yz = mix(ycc.yz, yccB.yz, clamp(chromaAmt * 1.55, 0.0, 1.0));

        // Mild luma smoothing (very small) to reduce patchiness without killing pores.
        float lumaAmt = foundStrength * (0.35 + 0.65 * skinLikely) * keepDetail * 0.14;
        ycc.x = mix(ycc.x, yccB.x, clamp(lumaAmt, 0.0, 1.0));

        // Gentle redness reduction: nudge Cr toward neutral in skin-likely regions.
        float redAmt = foundStrength * (0.35 + 0.65 * skinLikely) * keepDetail * 0.26;
        ycc.z = mix(ycc.z, 0.5, clamp(redAmt, 0.0, 1.0));

        color = yccToRgb(ycc);
        luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    }

    // Soft-glow finish: lift shadows/midtones slightly (avoid "matte" flattening).
    // Strongly reduced around edges/features by keepDetail.
    float mid = smoothstep(0.10, 0.50, luma) * (1.0 - smoothstep(0.62, 0.98, luma));
    float glow = foundStrength * keepDetail * (0.35 + 0.65 * skinLikely);
    float shadows = smoothstep(0.0, 0.60, 1.0 - luma);
    color += vec3(0.040) * glow * mid * shadows;

    // Slightly reduce saturation for a more "base" look, but keep it subtle (glowy, not dull).
    float sat = 1.0 - foundStrength * (0.40 + 0.60 * skinLikely) * 0.04;
    color = mix(vec3(luma), color, clamp(sat, 0.92, 1.0));

    // Foundation "finish": a tiny warm/neutral shift so the pass reads as base coverage
    // even when the source image is already smooth/low-noise.
    float finish = foundStrength * (0.35 + 0.65 * skinLikely) * keepDetail;
    color += vec3(0.015, 0.008, -0.004) * finish;
    luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Coverage/tint: make foundation read as actual base pigment.
    // Apply strongest in midtones, weaker in shadows/highlights to avoid a flat mask.
    float midCoverage = smoothstep(0.12, 0.30, luma) * (1.0 - smoothstep(0.62, 0.98, luma));
    float cov = uFoundCoverage
              * foundStrength
              * keepDetail
              * (0.45 + 0.55 * skinLikely)
              * midCoverage;
    color = mix(color, uFoundTint, clamp(cov, 0.0, 0.85));

    // ── Concealer (localized, optional) ──────────────────────────────────────
    // Intended for under-eye regions: reduce dark cast and lift shadows without
    // flattening highlights. Enabled per-draw via uniforms.
    float concealLift = concealLift0;
    float concealNeut = concealNeut0;
    if (concealLift > 0.001 || concealNeut > 0.001) {
        luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

        // Stronger in shadows/midtones; near-zero in highlights.
        float shadows = smoothstep(0.05, 0.70, 1.0 - luma);
        float mids = smoothstep(0.10, 0.55, luma) * (1.0 - smoothstep(0.60, 0.92, luma));
        float mask = (0.55 * shadows + 0.45 * mids) * (0.55 + 0.45 * skinLikely);

        // Boost where we actually detect a locally-dark patch (prevents “patchy” blobs).
        // patchMask already includes skinLikely + texKeep, so this stays subtle.
        mask *= (0.75 + 0.55 * patchMask);

        // Keep some texture even in concealer regions.
        float texKeep = 0.55 + 0.45 * keepDetail;

        if (concealNeut > 0.001) {
            vec3 ycc = rgbToYcc(color);
            ycc.yz = mix(ycc.yz, vec2(0.5), clamp(concealNeut * mask * texKeep, 0.0, 1.0));
            color = yccToRgb(ycc);
        }

        if (concealLift > 0.001) {
            // Lift by blending toward a soft white in shadows/mids only.
            float lift = clamp(concealLift * mask * texKeep, 0.0, 1.0);
            color = mix(color, color + vec3(0.14), lift);

            // Brown under-eye darkness can look gray when lifted; a tiny warm bias keeps it natural.
            // Applied only in the concealer path, and only where the lift mask is active.
            float warm = lift * 0.55;
            color += vec3(0.012, 0.006, -0.004) * warm;
        }
    }

    // Opacity boost: makes the foundation pass perceptible (camera → screen blend).
    // Keeps per-style control but avoids a “no visible change” outcome on-device.
    float foundAlpha = clamp(uFoundAlpha * 2.6, 0.0, 1.0);
    float autoAlpha  = clamp(uAutoCorrect * 3.0, 0.0, 1.0);
    // Concealer-only draws set uFoundAlpha/uAutoCorrect to 0, so include concealer strength
    // in the alpha budget or the pass will discard.
    float concealAlpha = clamp(max(uConcealLift, uConcealNeutralize) * 1.8, 0.0, 1.0);
    float finalAlpha = max(max(foundAlpha, autoAlpha), concealAlpha) * edgeAlpha;
    if (finalAlpha < 0.003) discard;

    if (uDebugFoundation > 2.5) {
        // Patch mask view: show detected dark-patch strength.
        float m = clamp(patchMask * 1.10, 0.0, 1.0);
        vec3 vis = mix(vec3(0.0), vec3(1.0), m);
        gl_FragColor = vec4(vis, finalAlpha);
    } else if (uDebugFoundation > 1.5) {
        // Delta view: amplify difference from the raw camera sample.
        vec3 d = abs(color - baseColor) * 12.0;
        gl_FragColor = vec4(clamp(d, 0.0, 1.0), finalAlpha);
    } else if (uDebugFoundation > 0.5) {
        gl_FragColor = vec4(uDebugColor, finalAlpha);
    } else {
        gl_FragColor = vec4(clamp(color, 0.0, 1.0), finalAlpha);
    }
}
