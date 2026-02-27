precision mediump float;

// Packed low-res mask (RGBA: contourCore, contourBlend, highlightCore, highlightBlend).
uniform sampler2D uMaskTex;

// Shared UV transform for consistency with other contour passes.
uniform mat3 uUvTransform;

// Skin-derived contour shade (sRGB 0..1), never pure black.
uniform vec3 uShadeRgb;

// Base strength (0..1) derived from style + lighting in Kotlin.
uniform float uBaseContour;

// Contrast boost multiplier (e.g. 1.20 for +20%).
uniform float uContrastBoost;

varying vec2 vTexCoord;

float saturate(float x) { return clamp(x, 0.0, 1.0); }

vec3 toLinearApprox(vec3 srgb) {
    return pow(max(srgb, vec3(0.0)), vec3(2.2));
}

vec3 toSrgbApprox(vec3 lin) {
    return pow(max(lin, vec3(0.0)), vec3(1.0 / 2.2));
}

void main() {
    vec2 uv = (uUvTransform * vec3(vTexCoord, 1.0)).xy;
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    vec4 m = texture2D(uMaskTex, uv);

    // Match the relight shader’s contour combination (core + blend).
    float mContour = saturate(m.r * 0.95 + m.g * 0.55);

    // Strength with requested contrast boost.
    float a = saturate(mContour * uBaseContour * max(uContrastBoost, 0.0));

    // Multiply factor (1.0 = no change, <1 darkens).
    // We construct the factor in an approximate-linear space, then convert back to sRGB
    // before the fixed-function GL multiply blends it with the framebuffer.
    float k = 0.78; // stronger: contour must read under typical phone exposure
    vec3 shadeSrgb = clamp(uShadeRgb, vec3(0.10), vec3(0.98));
    vec3 shadeLin = toLinearApprox(shadeSrgb);
    vec3 factorLin = mix(vec3(1.0), shadeLin, a * k);

    // Clamp to keep skin from going muddy/dirty in dim lighting.
    factorLin = clamp(factorLin, vec3(0.52), vec3(1.0));
    vec3 factor = clamp(toSrgbApprox(factorLin), vec3(0.0), vec3(1.0));

    gl_FragColor = vec4(clamp(factor, 0.0, 1.0), 1.0);
}

