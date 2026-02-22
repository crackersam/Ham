precision mediump float;

uniform vec4  uColor;         // RGBA makeup color (alpha pre-scaled by style opacity)
uniform float uBlendMode;     // 0 = normal, 1 = multiply-like
uniform float uGradientDir;   // 0 = flat, 1 = use gradient from inner->outer edge
uniform vec4  uColor2;        // secondary color for gradient (e.g. darker outer eyeshadow)

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

    float finalAlpha = baseColor.a * edgeAlpha;

    // Clamp to avoid artefacts on very transparent edges
    if (finalAlpha < 0.004) discard;

    gl_FragColor = vec4(rgb, finalAlpha);
}
