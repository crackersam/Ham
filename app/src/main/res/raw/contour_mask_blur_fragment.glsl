precision highp float;

uniform sampler2D uTex;
uniform vec2 uTexelSize; // 1/width, 1/height
uniform vec2 uDir;       // (1,0)=horizontal, (0,1)=vertical
uniform mat3 uUvTransform;

varying vec2 vTexCoord;

void main() {
    // Higher-quality separable Gaussian blur (9 taps).
    // We blur the *packed mask* only (RGBA), never the camera frame.
    vec2 uv = (uUvTransform * vec3(vTexCoord, 1.0)).xy;
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    vec2 o = uDir * uTexelSize;

    // Approximates sigma≈2.0. Running this 2× yields a very clean, patch-free mask.
    vec4 c = texture2D(uTex, uv) * 0.227027;
    c += (texture2D(uTex, clamp(uv + 1.0 * o, vec2(0.0), vec2(1.0))) +
          texture2D(uTex, clamp(uv - 1.0 * o, vec2(0.0), vec2(1.0)))) * 0.1945946;
    c += (texture2D(uTex, clamp(uv + 2.0 * o, vec2(0.0), vec2(1.0))) +
          texture2D(uTex, clamp(uv - 2.0 * o, vec2(0.0), vec2(1.0)))) * 0.1216216;
    c += (texture2D(uTex, clamp(uv + 3.0 * o, vec2(0.0), vec2(1.0))) +
          texture2D(uTex, clamp(uv - 3.0 * o, vec2(0.0), vec2(1.0)))) * 0.054054;
    c += (texture2D(uTex, clamp(uv + 4.0 * o, vec2(0.0), vec2(1.0))) +
          texture2D(uTex, clamp(uv - 4.0 * o, vec2(0.0), vec2(1.0)))) * 0.016216;
    gl_FragColor = c;
}

