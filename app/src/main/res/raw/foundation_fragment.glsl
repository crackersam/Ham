#extension GL_OES_EGL_image_external : require
precision mediump float;

uniform samplerExternalOES uTexture;
uniform vec2  uTexelSize;
uniform float uSmooth;      // skin-smoothing intensity (renderer-level, 0–1)
uniform float uTone;        // warmth intensity, kept in sync with camera shader
uniform float uFoundAlpha;  // per-style foundation opacity (0–1)

varying float vEdgeFactor;
varying vec2  vCamUV;

// 5-tap weighted cross blur — samples camera OES texture directly
vec3 smoothSkin(vec2 uv) {
    vec2 t = uTexelSize * 1.5;
    vec3 c = texture2D(uTexture, uv).rgb;
    vec3 n = texture2D(uTexture, uv + vec2( 0.0,  t.y)).rgb;
    vec3 s = texture2D(uTexture, uv + vec2( 0.0, -t.y)).rgb;
    vec3 e = texture2D(uTexture, uv + vec2( t.x,  0.0)).rgb;
    vec3 w = texture2D(uTexture, uv + vec2(-t.x,  0.0)).rgb;
    return (c * 4.0 + n + s + e + w) / 8.0;
}

void main() {
    // Two-stage feather: same curve as makeup_fragment so all layers blend
    // consistently — sharper near-centre ramp, longer gentle outer fade
    float edgeAlpha = smoothstep(0.0, 0.20, vEdgeFactor)
                    * smoothstep(0.0, 0.55, vEdgeFactor);

    vec3 color = texture2D(uTexture, vCamUV).rgb;
    float luma  = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Luminance-gated skin smoothing (avoids blurring dark brows/lashes)
    if (uSmooth > 0.01) {
        vec3 blurred    = smoothSkin(vCamUV);
        float skinLikely = smoothstep(0.20, 0.45, luma)
                         * smoothstep(0.92, 0.65, luma);
        color = mix(color, blurred, uSmooth * skinLikely * 0.80);
        // Recalculate luma after smoothing for tone pass below
        luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    }

    // Subtle warmth + saturation boost — kept identical to camera_fragment.glsl
    // so the foundation surface blends seamlessly with the unfiltered background
    if (uTone > 0.01) {
        color = mix(vec3(luma), color, 1.0 + uTone * 0.12);
        color += vec3(0.018, 0.010, 0.004) * uTone
                 * smoothstep(0.3, 0.65, luma);
    }

    float finalAlpha = uFoundAlpha * edgeAlpha;
    if (finalAlpha < 0.004) discard;

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), finalAlpha);
}
