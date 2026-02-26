precision mediump float;

uniform sampler2D uMaskTex;
uniform sampler2D uFrameTex; // base frame reference (soft-light + luminance-driven multiply)
uniform vec4 uShade;         // rgb = contour shade, a = intensity
uniform float uBlendMode;    // 0 = multiply-factor, 1 = soft-light
uniform float uPass;         // 0 = contour, 1 = highlight

varying vec2 vTexCoord;

vec3 softLight(vec3 base, vec3 blend) {
    vec3 r1 = base - (1.0 - 2.0 * blend) * base * (1.0 - base);
    vec3 d0 = ((16.0 * base - 12.0) * base + 4.0) * base;
    vec3 d1 = sqrt(base);
    vec3 d  = mix(d0, d1, step(vec3(0.25), base));
    vec3 r2 = base + (2.0 * blend - 1.0) * (d - base);
    return mix(r1, r2, step(vec3(0.5), blend));
}

void main() {
    vec4 ms = texture2D(uMaskTex, vTexCoord);
    float mContour = ms.r;
    float mHighlight = ms.g;
    float baseIntensity = clamp(uShade.a, 0.0, 1.0);
    float contourAlpha = clamp(mContour * baseIntensity, 0.0, 1.0);
    vec3 under = texture2D(uFrameTex, vTexCoord).rgb;

    // Requirement: slightly desaturate the contour shade.
    vec3 contourShade = clamp(uShade.rgb, 0.0, 1.0);
    float shadeLum = dot(contourShade, vec3(0.2126, 0.7152, 0.0722));
    contourShade = mix(contourShade, vec3(shadeLum), 0.15);

    // Highlight pass: subtle lift above the contour (screen blend in fixed-function).
    if (uPass > 0.5) {
        // Final micro-polish:
        // - Slightly stronger highlight alpha (0.12 -> 0.16 max)
        // - Slight edge sharpening on alpha
        float hlStrength = clamp(baseIntensity * 0.32, 0.08, 0.16);
        float hlA = clamp(mHighlight * hlStrength, 0.0, 1.0);
        hlA = pow(hlA, 0.8);

        // Lighter version of the local skin tone (not white): derive from frame sample.
        // Requirement: highlightColor = baseColor * 1.08
        vec3 hlColor = clamp(under * 1.08, 0.0, 1.0);
        float hlLum = dot(hlColor, vec3(0.2126, 0.7152, 0.0722));
        hlColor = mix(hlColor, vec3(hlLum), 0.08);

        // For screen blending via glBlendFunc(ONE, ONE_MINUS_SRC_COLOR),
        // output the highlight "source" color scaled by alpha.
        // Add a very subtle specular touch near the top edge of the highlight.
        float edge = smoothstep(0.6, 1.0, mHighlight);
        vec3 outRgb = hlColor * hlA + vec3(edge * 0.03) * hlA;
        outRgb = clamp(outRgb, 0.0, 1.0);
        gl_FragColor = vec4(outRgb, hlA);
        return;
    }

    if (uBlendMode < 0.5) {
        // Multiply fallback (safe, avoid over-darkening):
        // finalRgb = baseRgb * f, where f in [0,1]
        float f = 1.0 - contourAlpha * 0.55;

        // FIX 1 — Boost local micro-contrast only in contour region.
        // Keep it natural: do not allow the contrast stage to darken further.
        float contrast = 1.08;
        vec3 contrasted = (under - 0.5) * contrast + 0.5;
        vec3 contrastMix = mix(under, contrasted, contourAlpha * 0.4);
        vec3 denom = max(under, vec3(1e-4));
        vec3 contrastFactor = max(contrastMix / denom, vec3(1.0));

        vec3 factor = clamp(vec3(f) * contrastFactor, 0.0, 1.0);
        gl_FragColor = vec4(factor, contourAlpha);
    } else {
        // SOFT_LIGHT: output the shaded color (alpha blend composes onto the current framebuffer).
        vec3 shaded = softLight(under, contourShade);

        // FIX 1 — Boost local micro-contrast only in contour region.
        // Keep it natural: do not allow the contrast stage to darken further.
        float contrast = 1.08;
        vec3 contrasted = (shaded - 0.5) * contrast + 0.5;
        vec3 contrastedSafe = max(contrasted, shaded);
        vec3 outRgb = mix(shaded, contrastedSafe, contourAlpha * 0.4);
        outRgb = clamp(outRgb, 0.0, 1.0);
        gl_FragColor = vec4(outRgb, contourAlpha);
    }
}

