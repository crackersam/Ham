precision highp float;

uniform sampler2D uFrameTex;
uniform sampler2D uMaskTex; // RGBA: cheek, jaw, nose, chin

// Shared UV transform for all passes (identity unless overridden).
uniform mat3 uUvTransform;

// Skin-derived contour shade (already computed from sampled skin tone in Kotlin).
uniform vec3 uSkinRgb;      // contour shade (sRGB 0..1), never pure black
uniform float uCoolTone;    // 0..1 (optional extra cool shift)

uniform vec4 uIntensity;    // per-region strengths (cheek, jaw, nose, chin)
uniform float uMaster;      // overall (style alpha etc)

uniform float uBlendMode;   // 0 = multiply, 1 = softlight

// Pipeline debug stage:
// 0 = final composite
// 1 = raw camera/base frame (uFrameTex)
// 2 = raw contour mask (grayscale; bound in uMaskTex)
// 3 = blurred mask (grayscale; bound in uMaskTex)
uniform float uStage;

// Debug:
// 0 = normal
// 1 = mask cheek
// 2 = mask jaw
// 3 = mask nose
// 4 = mask chin
// 5 = mask combined strength
uniform float uDebugMode;

varying vec2 vTexCoord;

float saturate(float x) { return clamp(x, 0.0, 1.0); }

float hash12(vec2 p) {
    // Deterministic, cheap screen-space hash (0..1).
    // Using gl_FragCoord makes it stable across UV jitter and avoids “swimming” noise.
    float h = dot(p, vec2(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

vec3 softLight(vec3 base, vec3 blend) {
    vec3 r1 = base - (1.0 - 2.0 * blend) * base * (1.0 - base);
    vec3 d0 = ((16.0 * base - 12.0) * base + 4.0) * base;
    vec3 d1 = sqrt(base);
    vec3 d  = mix(d0, d1, step(vec3(0.25), base));
    vec3 r2 = base + (2.0 * blend - 1.0) * (d - base);
    return mix(r1, r2, step(vec3(0.5), blend));
}

void main() {
    vec2 uv = (uUvTransform * vec3(vTexCoord, 1.0)).xy;
    uv = clamp(uv, vec2(0.0), vec2(1.0));

    if (uStage > 0.5) {
        if (uStage < 1.5) {
            vec3 base0 = texture2D(uFrameTex, uv).rgb;
            gl_FragColor = vec4(base0, 1.0);
            return;
        } else {
            vec4 m0 = texture2D(uMaskTex, uv);
            float v = max(max(m0.r, m0.g), max(m0.b, m0.a));
            gl_FragColor = vec4(vec3(saturate(v)), 1.0);
            return;
        }
    }

    vec3 base = texture2D(uFrameTex, uv).rgb;
    vec4 m = texture2D(uMaskTex, uv);

    float cheek = m.r;
    float jaw   = m.g;
    float nose  = m.b;
    float chin = m.a;

    if (uDebugMode > 0.5) {
        float v = 0.0;
        if (uDebugMode < 1.5) v = cheek;
        else if (uDebugMode < 2.5) v = jaw;
        else if (uDebugMode < 3.5) v = nose;
        else if (uDebugMode < 4.5) v = chin;
        else v = saturate(cheek * uIntensity.x + jaw * uIntensity.y + nose * uIntensity.z + chin * uIntensity.w);
        gl_FragColor = vec4(vec3(v), 1.0);
        return;
    }

    float master = saturate(uMaster);
    vec4 k = clamp(uIntensity, 0.0, 1.0) * master;

    float strength =
        cheek  * k.x +
        jaw    * k.y +
        nose   * k.z +
        chin   * k.w;
    strength = saturate(strength);

    // Tiny, gated dither to suppress 8-bit banding in smooth gradients.
    // Keep it strictly inside the mask so it never “leaks” into background.
    {
        float gate = smoothstep(0.02, 0.18, strength);
        float n = hash12(gl_FragCoord.xy);
        float d = (n - 0.5) * (1.0 / 255.0);
        strength = saturate(strength + d * gate);
    }

    // Anti-flatness boost (inside contour mask only):
    // slightly increase local contrast + slightly darken midtones under the mask,
    // then apply the shadow multiply.
    {
        float c = 0.18 * strength;     // contrast amount
        float d = 0.06 * strength;     // darken amount
        vec3 contrasted = clamp((base - 0.5) * (1.0 + c) + 0.5, 0.0, 1.0);
        contrasted *= (1.0 - d);
        base = mix(base, contrasted, strength);
    }

    // Shade is computed from sampled skin tone (not pure black).
    vec3 shade = clamp(uSkinRgb, vec3(0.04), vec3(0.98));
    // Optional extra cool shift (very small; keeps shade stable across devices).
    float ct = clamp(uCoolTone, 0.0, 1.0);
    shade.r *= mix(1.0, 0.96, ct);
    shade.g *= mix(1.0, 0.98, ct);
    shade.b *= mix(1.0, 1.01, ct);

    vec3 outRgb;
    if (uBlendMode < 0.5) {
        // Multiply-like: base * mix(1, shade, strength)
        vec3 factor = mix(vec3(1.0), shade, strength);
        outRgb = base * factor;
    } else {
        // Premium: soft-light, then mix by strength.
        vec3 sl = softLight(base, shade);
        outRgb = mix(base, sl, strength);
    }

    gl_FragColor = vec4(clamp(outRgb, 0.0, 1.0), 1.0);
}

