precision mediump float;

uniform vec4  uColor;         // RGBA makeup color (alpha pre-scaled by style opacity)
uniform float uBlendMode;     // 0 = normal, 1 = multiply-like
uniform float uGradientDir;   // 0 = flat, 1 = use gradient from inner->outer edge
uniform vec4  uColor2;        // secondary color for gradient (e.g. darker outer eyeshadow)
uniform float uEffectKind;    // 0 = default, 1 = blush, 2 = sparkle, 3 = contour(shadow)
uniform sampler2D uNoiseTex;  // small repeatable noise/grain texture
uniform float uNoiseScale;    // tiling in region UV space
uniform float uNoiseAmount;   // 0..1 subtle alpha modulation
uniform float uTime;          // seconds, for gentle twinkle (sparkle effect)

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
    vec4 baseColor = uColor;
    if (uEffectKind < 2.5 && uGradientDir > 0.5) {
        // Eyeshadow: gradient from inner (uColor) to outer (uColor2)
        // vRegionUV.x = 0 at inner corner, 1 at outer corner
        float t = smoothstep(0.0, 1.0, vRegionUV.x);
        baseColor = mix(uColor, uColor2, t);
    }

    // ── Multiply blend approximation ────────────────────────────────────────
    // For lips/liner we want multiply-like darkening rather than plain alpha.
    // We approximate with a darkened composite: output = color * (1-factor) + color*base*factor
    // Shader handles alpha; the actual GL blend equation does the rest.
    vec3 rgb = baseColor.rgb;
    if (uEffectKind < 2.5 && uBlendMode > 0.5) {
        // Darken toward the makeup color (simulate multiply)
        rgb = baseColor.rgb * (0.65 + 0.35 * (1.0 - edgeAlpha));
    }

    // ── Procedural effect paths ──────────────────────────────────────────────
    // uEffectKind:
    //  0 = default
    //  1 = blush realism (radial falloff + subtle grain)
    //  2 = sparkle (sparse glitter + gentle twinkle)
    //  3 = contour (tinted shadow factor for multiply compositing)
    //  4 = highlight (softer falloff + micro-dither to avoid visible edge lines)
    if (uEffectKind > 3.5 && uEffectKind < 4.5) {
        // ── Highlight (pro finish) path ──────────────────────────────────────
        // The default discard cutoff can create a visible contour at very low
        // alpha. Highlights benefit from a longer tail + micro-dither so the
        // fade remains imperceptible.
        float a0 = smoothstep(0.0, 0.14, vEdgeFactor) * smoothstep(0.0, 0.95, vEdgeFactor);
        a0 = pow(a0, 0.85); // slightly longer tail

        // Micro-dither near the rim to break up coherent “edge lines”.
        float rim = 1.0 - smoothstep(0.30, 0.88, vEdgeFactor); // 1 at rim → 0 toward center
        if (uNoiseAmount > 0.001) {
            vec2 duv = vec2(vRegionUV.x, vEdgeFactor) * uNoiseScale;
            float n = texture2D(uNoiseTex, duv).r; // 0..1
            float d = (n - 0.5) * uNoiseAmount;    // small signed
            a0 = clamp(a0 + d * 0.14 * rim, 0.0, 1.0);
        }

        edgeAlpha = a0;
    } else if (uEffectKind > 2.5 && uEffectKind < 3.5) {
        // ── Contour (pro shadow) path ────────────────────────────────────────
        // Outputs an RGB *factor* suitable for GL_DST_COLOR multiply blending.
        //
        // - vEdgeFactor provides feathering (soft edges)
        // - vRegionUV provides a region-local coordinate for shaping the mask:
        //     * ellipse meshes: radial + "upper/outer" bias via vRegionUV.y
        //     * ribbon meshes: vRegionUV.y should be 1.0 at the strong edge and 0.0 at the fade edge
        vec2  p  = vRegionUV - vec2(0.5);
        float rn = clamp(length(p) * 2.0, 0.0, 1.0); // 0=center .. 1=rim

        // Tighter gaussian than blush; keeps a defined sculpt without harsh edges.
        float gaussian = exp(-rn * rn * 5.6);

        // Shape hint via uGradientDir (ignored outside contour mode):
        //  0 = ellipse-style regionUV (use gaussian)
        //  2 = ribbon/stroke regionUV (skip radial gaussian so the shadow doesn't fade at the ends)
        float ribbonShape = step(1.5, uGradientDir);
        gaussian = mix(gaussian, 1.0, ribbonShape);

        // Bias toward the "upper/outer" part of the region.
        float biasT = smoothstep(0.35, 0.92, vRegionUV.y);
        float directional = mix(0.72, 1.00, biasT);

        float mask = edgeAlpha * gaussian * directional;

        // Very subtle grain to avoid banding / sticker-flatness.
        vec2 nuv = vRegionUV * uNoiseScale;
        float n = texture2D(uNoiseTex, nuv).r; // 0..1
        float grain = 1.0 + (n - 0.5) * uNoiseAmount;
        mask *= clamp(grain, 0.88, 1.12);

        float s = clamp(baseColor.a * mask, 0.0, 1.0);

        // Multiply-factor output: 1.0 = no darkening, lower = deeper shadow.
        rgb = mix(vec3(1.0), baseColor.rgb, s);

        // Keep alpha meaningful for capture/readback and alpha-separate blending.
        edgeAlpha = mask;
    } else if (uEffectKind > 0.5 && uEffectKind < 1.5) {
        // ── Blush realism path ───────────────────────────────────────────────
        // Uses radial falloff (region UV) + subtle grain modulation so blush
        // doesn't look like a perfectly-uniform sticker.
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
    } else if (uEffectKind > 1.5) {
        // ── Sparkle / glitter ────────────────────────────────────────────────
        // Build a sparse glitter mask from the noise texture and modulate it
        // over time.  Use a 2D seed that works even for fan meshes where
        // vRegionUV.y can be constant on the rim.
        float intensity = clamp(uNoiseAmount, 0.0, 1.0);
        vec2 suv = vec2(vRegionUV.x, vEdgeFactor) * uNoiseScale;

        // Two taps with a time offset keep the twinkle organic without needing
        // a second texture.
        float n0 = texture2D(uNoiseTex, suv).r; // 0..1
        float n1 = texture2D(uNoiseTex, suv + vec2(3.7, 1.9) + vec2(uTime * 0.35, uTime * 0.17)).r;
        // Use max so peaks survive GL_LINEAR filtering (averaging makes near-1 rare).
        float n = max(n0, n1);

        // Higher intensity -> lower threshold -> more sparkles.
        float baseThresh = mix(0.86, 0.62, intensity);

        // Gentle twinkle: shift threshold a little per-fragment.
        float tw = 0.5 + 0.5 * sin(uTime * 5.5 + n0 * 18.0);
        float thresh = baseThresh + (tw - 0.5) * 0.07;

        float sparkle = smoothstep(thresh, 1.0, n);
        sparkle = sparkle * sparkle; // sharpen

        // Keep sparkles mostly in the denser interior of the mesh.
        float inner = smoothstep(0.06, 0.55, vEdgeFactor);
        edgeAlpha = sparkle * inner;
    }

    float finalAlpha = baseColor.a * edgeAlpha;

    // Clamp to avoid artefacts on very transparent edges
    float cutoff = 0.004;
    if (uEffectKind > 1.5 && uEffectKind < 2.5) cutoff = 0.001;
    // Highlights need the full alpha tail to stay imperceptible at the rim.
    // Discarding even tiny alphas can create a visible contour line.
    if (uEffectKind > 3.5 && uEffectKind < 4.5) cutoff = 0.0;
    if (finalAlpha < cutoff) discard;

    gl_FragColor = vec4(rgb, finalAlpha);
}
