precision mediump float;

uniform sampler2D uTexture;
varying vec2 vTexCoord;

// Skin smoothing has moved to foundation_fragment.glsl so it is applied
// only within the face-oval mesh boundary, not across the whole frame.
// This pass outputs the raw camera image with a subtle global warmth tone.
uniform float uTone;  // 0–1 subtle warmth

void main() {
    vec3 color = texture2D(uTexture, vTexCoord).rgb;

    // Subtle warmth + slight saturation boost (global scene base)
    if (uTone > 0.01) {
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(luma), color, 1.0 + uTone * 0.12);
        color += vec3(0.018, 0.010, 0.004) * uTone
                 * smoothstep(0.3, 0.65, luma);
    }

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
