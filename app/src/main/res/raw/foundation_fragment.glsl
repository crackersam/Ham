precision mediump float;

uniform sampler2D uTexture;
uniform vec2  uTexelSize;
uniform float uSmooth;      // skin-smoothing intensity (renderer-level, 0–1)
uniform float uTone;        // warmth intensity, kept in sync with camera shader
uniform float uFoundAlpha;  // per-style foundation opacity (0–1)
uniform vec3  uFoundTint;   // foundation pigment tint (linear-ish RGB)
uniform float uFoundCoverage; // 0..1 coverage strength (tint blend)
uniform float uDebugFoundation; // 0 = normal, 1 = debug face mask
uniform vec3  uDebugColor;      // debug mask tint color

varying float vEdgeFactor;
varying vec2  vCamUV;

// Weighted cross blur — samples camera texture directly
vec3 smoothSkin(vec2 uv) {
    // Larger radius so the reference differs enough to be perceptible at phone resolutions,
    // while keeping the tap count low for compatibility/perf.
    vec2 t = uTexelSize * 4.2;
    vec3 c = texture2D(uTexture, uv).rgb;
    vec3 n = texture2D(uTexture, uv + vec2( 0.0,  t.y)).rgb;
    vec3 s = texture2D(uTexture, uv + vec2( 0.0, -t.y)).rgb;
    vec3 e = texture2D(uTexture, uv + vec2( t.x,  0.0)).rgb;
    vec3 w = texture2D(uTexture, uv + vec2(-t.x,  0.0)).rgb;
    return (c * 4.0 + n + s + e + w) / 8.0;
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

    // Skin likelihood gate (midtone bias; keeps non-skin + highlights less affected).
    // Loosened to still apply in darker indoor lighting.
    float skinLikely = smoothstep(0.12, 0.40, luma)
                     * smoothstep(0.95, 0.55, luma);

    // Local-detail mask: if blur diverges from original, it's likely an edge/texture.
    // Reduce correction there to keep features crisp (brows/lashes).
    float detail = length(color - blurred);               // ~0..0.3 typical
    float keepDetail = 1.0 - smoothstep(0.03, 0.14, detail);

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
    float chromaKeep = 1.0 - smoothstep(0.06, 0.24, detail);
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
    float mid = smoothstep(0.10, 0.50, luma) * smoothstep(0.98, 0.62, luma);
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
    float midCoverage = smoothstep(0.12, 0.30, luma) * smoothstep(0.98, 0.62, luma);
    float cov = uFoundCoverage
              * foundStrength
              * keepDetail
              * (0.45 + 0.55 * skinLikely)
              * midCoverage;
    color = mix(color, uFoundTint, clamp(cov, 0.0, 0.85));

    // Opacity boost: makes the foundation pass perceptible (camera → screen blend).
    // Keeps per-style control but avoids a “no visible change” outcome on-device.
    float finalAlpha = clamp(uFoundAlpha * 2.6, 0.0, 1.0) * edgeAlpha;
    if (finalAlpha < 0.004) discard;

    if (uDebugFoundation > 1.5) {
        // Delta view: amplify difference from the raw camera sample.
        vec3 d = abs(color - baseColor) * 12.0;
        gl_FragColor = vec4(clamp(d, 0.0, 1.0), finalAlpha);
    } else if (uDebugFoundation > 0.5) {
        gl_FragColor = vec4(uDebugColor, finalAlpha);
    } else {
        gl_FragColor = vec4(clamp(color, 0.0, 1.0), finalAlpha);
    }
}
