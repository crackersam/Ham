precision mediump float;

uniform vec4  uColor;         // RGBA makeup color (alpha pre-scaled by style opacity)
uniform float uBlendMode;     // 0 = normal, 1 = multiply-like
uniform float uGradientDir;   // 0 = flat, 1 = use gradient from inner->outer edge
uniform vec4  uColor2;        // secondary color for gradient (e.g. darker outer eyeshadow)
uniform float uEffectKind;    // 0 = default, 1 = blush
uniform sampler2D uNoiseTex;  // small repeatable noise/grain texture
uniform float uNoiseScale;    // tiling in region UV space
uniform float uNoiseAmount;   // 0..1 subtle alpha modulation

varying float vEdgeFactor;   // 0=edge → 1=center, drives feathering
varying vec2  vRegionUV;     // [0,1] within region

void main() {
    // ── Feathered alpha ──────────────────────────────────────────────────────
    // Two-stage smoothstep product: sharper inner ramp (0→0.20) keeps pigment
    // dense near the centre; the outer ramp (0→0.55) gives a longer, gentler
    // fade so edges blend naturally — matching professional / TikTok-filter
    // quality rather than a blunt cutoff.
    float edgeAlpha = smoothstep(0.0, 0.20, vEdgeFactor)
                    * smoothstep(0.0, 0.55, vEdgeFactor);

    // ── Color / gradient ────────────────────────────────────────────────────
    vec4 baseColor;
    if (uGradientDir > 0.5) {
        // Eyeshadow: gradient from inner (uColor) to outer (uColor2)
        // vRegionUV.x = 0 at inner corner, 1 at outer corner
        float t = smoothstep(0.0, 1.0, vRegionUV.x);
        baseColor = mix(uColor, uColor2, t);
    } else {
        baseColor = uColor;
    }

    // ── Multiply blend approximation ────────────────────────────────────────
    // For lips/liner we want multiply-like darkening rather than plain alpha.
    // We approximate with a darkened composite: output = color * (1-factor) + color*base*factor
    // Shader handles alpha; the actual GL blend equation does the rest.
    vec3 rgb = baseColor.rgb;
    if (uBlendMode > 0.5) {
        // Darken toward the makeup color (simulate multiply)
        rgb = baseColor.rgb * (0.65 + 0.35 * (1.0 - edgeAlpha));
    }

    // ── Blush realism path ───────────────────────────────────────────────────
    // Uses radial falloff (region UV) + subtle grain modulation so blush
    // doesn't look like a perfectly-uniform sticker.
    if (uEffectKind > 0.5) {
        vec2  p  = vRegionUV - vec2(0.5);
        float rn = clamp(length(p) * 2.0, 0.0, 1.0); // 0=center .. 1=rim

        // Gaussian-like pigment distribution with a gentle rim feather.
        float gaussian = exp(-rn * rn * 4.2);
        float rimFeather = smoothstep(0.0, 0.28, vEdgeFactor);
        float blushMask = gaussian * rimFeather;

        // Mild desaturation reads more like real pigment on skin.
        float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        rgb = mix(vec3(luma), rgb, 0.92);

        // Grain: modulate alpha slightly with repeatable region-anchored noise.
        // uNoiseAmount is expected to be small (e.g. 0.08–0.16).
        vec2 nuv = vRegionUV * uNoiseScale;
        float n = texture2D(uNoiseTex, nuv).r; // 0..1
        float grain = 1.0 + (n - 0.5) * uNoiseAmount;
        blushMask *= clamp(grain, 0.75, 1.25);

        edgeAlpha = blushMask;
    }

    float finalAlpha = baseColor.a * edgeAlpha;

    // Clamp to avoid artefacts on very transparent edges
    if (finalAlpha < 0.004) discard;

    gl_FragColor = vec4(rgb, finalAlpha);
}
