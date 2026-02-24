precision mediump float;

uniform vec4  uColor;         // RGBA makeup color (alpha pre-scaled by style opacity)
uniform float uBlendMode;     // 0 = normal, 1 = multiply-like
uniform float uGradientDir;   // 0 = flat, 1 = use gradient from inner->outer edge
uniform vec4  uColor2;        // secondary color for gradient (e.g. darker outer eyeshadow)
uniform float uEffectKind;    // 0 = default, 1 = blush, 2 = sparkle, 3 = contour(shadow)
uniform sampler2D uCameraTex; // camera texture (RGBA) for per-pixel adaptation
uniform float uMirror;        // 1.0 for mirror preview, 0.0 for unmirrored
uniform float uSkinLuma;      // estimated skin luma (0..1), used to suppress beard/shadow artefacts
uniform sampler2D uNoiseTex;  // small repeatable noise/grain texture
uniform float uNoiseScale;    // tiling in region UV space
uniform float uNoiseAmount;   // 0..1 subtle alpha modulation
uniform float uTime;          // seconds, for gentle twinkle (sparkle effect)

varying float vEdgeFactor;   // 0=edge → 1=center, drives feathering
varying vec2  vRegionUV;     // [0,1] within region
varying vec2  vNdcPos;       // clip-space xy after cropScale ([-1,1])

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
        // Cheek contour should not form a ring with dark borders. Instead of a
        // two-edge "band/ring" mask, use a single smooth hump (2D gaussian)
        // shifted toward the cheekbone side so it blends like a pro application.
        vec2 q = (vRegionUV - vec2(0.5)) * 2.0; // -1..1 in ellipse-local space
        // Along-axis softness: prevent a visible diagonal “stripe” that can read
        // as a line from outer eye to mouth corner on some faces.
        //
        // Strongly fade at the *ends* of the ellipse while keeping a soft, blended
        // center section.
        float ax = abs(q.x);
        float endFade = 1.0 - smoothstep(0.42, 1.05, ax);
        endFade = pow(clamp(endFade, 0.0, 1.0), 1.15);
        float gx = exp(-q.x * q.x * 1.70) * endFade;
        // Across-axis: shift upward so the darkest part sits under the cheekbone, then fades.
        float y0 = 0.32; // +Y = "up" in ellipse local space (renderer enforces axisY up)
        float gy = exp(-(q.y - y0) * (q.y - y0) * 3.15);
        float ellipseShape = pow(clamp(gx * gy, 0.0, 1.0), 0.92);

        // Shape hint via uGradientDir (ignored outside contour mode):
        //  0 = ellipse-style regionUV (use gaussian)
        //  2 = ribbon/stroke regionUV (skip radial gaussian so the shadow doesn't fade at the ends)
        float ribbonShape = step(1.5, uGradientDir);
        float shape = mix(ellipseShape, 1.0, ribbonShape);

        // Bias toward the "upper/outer" part of the region.
        float biasT = smoothstep(0.28, 0.96, vRegionUV.y);
        biasT = pow(clamp(biasT, 0.0, 1.0), 1.30);
        float directional = mix(0.62, 1.00, biasT);

        // For ellipse regions, also bias toward the "outer" side along the local +X axis.
        // (For ribbon/stroke meshes vRegionUV.x is progress, so keep it neutral.)
        float alongT = smoothstep(0.22, 0.94, vRegionUV.x);
        float along = mix(0.80, 1.00, alongT);
        along = mix(along, 1.00, ribbonShape);

        // Contour edge handling:
        // Use a longer alpha tail than regular makeup so the shadow "melts" into skin
        // without a visible boundary line on any skin tone.
        float edgeA = smoothstep(0.0, 0.38, vEdgeFactor) * smoothstep(0.0, 0.98, vEdgeFactor);
        edgeA = pow(clamp(edgeA, 0.0, 1.0), 0.75);

        float mask = edgeA * shape * directional * along;

        // Per-pixel adaptation (crucial for men + all skin tones):
        // Suppress contour where the underlying pixel is *significantly darker than skin*
        // and low-saturation (beard/stubble shadow or deep cast shadow), which otherwise
        // reads as a grey stripe when multiplied.
        vec2 uv = vec2(vNdcPos.x * 0.5 + 0.5, 0.5 - 0.5 * vNdcPos.y);
        if (uMirror > 0.5) uv.x = 1.0 - uv.x;
        vec3 under = texture2D(uCameraTex, uv).rgb;
        float underLuma = dot(under, vec3(0.2126, 0.7152, 0.0722));
        float mx = max(under.r, max(under.g, under.b));
        float mn = min(under.r, min(under.g, under.b));
        float underSat = mx - mn;
        float darkDelta = clamp(uSkinLuma - underLuma, 0.0, 0.25);
        float beard = smoothstep(0.05, 0.14, darkDelta) * (1.0 - smoothstep(0.10, 0.30, underSat));
        mask *= mix(1.0, 0.45, beard);

        // Very subtle grain to avoid banding / sticker-flatness.
        vec2 nuv = vRegionUV * uNoiseScale;
        float n = texture2D(uNoiseTex, nuv).r; // 0..1
        float grain = 1.0 + (n - 0.5) * uNoiseAmount;
        mask *= clamp(grain, 0.88, 1.12);

        // Micro-dither: breaks up faint contour banding/edge lines without reading as grain.
        // Strongest near the edge where banding is most visible.
        if (uNoiseAmount > 0.001) {
            float rim = 1.0 - smoothstep(0.32, 0.88, vEdgeFactor);
            float d = (n - 0.5) * uNoiseAmount;
            mask = clamp(mask + d * 0.08 * rim, 0.0, 1.0);
        }

        // Strength curve:
        // Keep response close to linear so contour never turns into a painted stripe.
        float s0 = clamp(baseColor.a * mask, 0.0, 1.0);
        float s = pow(s0, 1.10);

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

    // Clamp to avoid artefacts on very transparent edges.
    //
    // Important: for contour/highlight we intentionally keep the full alpha tail
    // so the layer blends out cleanly. Discarding tiny alphas is a common cause
    // of visible “edge lines” on soft makeup.
    float cutoff = 0.004;
    if (uEffectKind > 1.5 && uEffectKind < 2.5) cutoff = 0.001; // sparkle
    if (uEffectKind > 2.5 && uEffectKind < 3.5) cutoff = 0.0;   // contour
    if (uEffectKind > 3.5 && uEffectKind < 4.5) cutoff = 0.0;   // highlight
    if (finalAlpha < cutoff) discard;

    gl_FragColor = vec4(rgb, finalAlpha);
}
