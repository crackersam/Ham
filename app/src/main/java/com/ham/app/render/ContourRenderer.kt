package com.ham.app.render

import android.content.Context
import androidx.compose.ui.graphics.Color
import com.ham.app.data.FACE_OVAL
import kotlin.math.max
import kotlin.math.min

/**
 * Facade for contour/highlight rendering.
 *
 * Owns the underlying GPU effect (mask → blur → multiply/softlight composite) and maps
 * style intent + lighting metrics into TikTok/Snap-like visible sculpting.
 */
class ContourRenderer(context: Context) {
    private val effect = ContourMakeupEffect(context)

    /**
     * User-tunable controls (0..1):
     * - [contourIntensity] is a global multiplier (in addition to style.contourAlpha)
     * - per-region multipliers control relative placement emphasis
     *
     * Defaults match the requested starting point.
     */
    var contourIntensity: Float = 1.0f
    var cheekMultiplier: Float = 0.48f
    var jawMultiplier: Float = 0.34f
    var templeMultiplier: Float = 0.28f
    var noseMultiplier: Float = 0.18f

    // EMA smoothing for per-frame computed strengths (prevents pumping).
    private var hasStrengthEma = false
    private var cheekEma = 0f
    private var jawEma = 0f
    private var noseEma = 0f
    private var foreheadEma = 0f
    private var baseContourEma = 0f

    fun isReady(): Boolean = effect.isReady()

    fun onSurfaceCreated() {
        effect.onSurfaceCreated()
    }

    fun onSurfaceChanged(viewWidth: Int, viewHeight: Int) {
        effect.onSurfaceChanged(viewWidth, viewHeight)
    }

    fun setCropScale(scaleX: Float, scaleY: Float) {
        effect.setCropScale(scaleX, scaleY)
    }

    fun setUvTransform3x3(m: FloatArray) {
        effect.setUvTransform3x3(m)
    }

    data class LightingMetrics(
        val faceMeanY: Float,
        val faceStdY: Float,
        val clipFrac: Float,
        val lightBias: Float,
    )

    data class Strengths(
        val cheek: Float,
        val jaw: Float,
        val nose: Float,
        val forehead: Float,
        val baseContour: Float,
        val baseHighlight: Float,
        val baseMicro: Float,
        val baseSpec: Float,
    )

    data class BuiltContour(
        val maskTex: Int,
        val shade: Color,
        val baseContour: Float,
    )

    /**
     * Computes per-region strengths (0..1) from style + face size + lighting.
     *
     * Requirements:
     * - If face is small → increase slightly
     * - If lighting is low → reduce harshness
     */
    fun computeStrengths(
        lm: FloatArray,
        style: com.ham.app.data.MakeupStyle,
        lighting: LightingMetrics,
    ): Strengths {
        val faceWidthUv = run {
            var minX = 1f
            var maxX = 0f
            for (idx in FACE_OVAL) {
                val x = lm[idx * 3].coerceIn(0f, 1f)
                if (x < minX) minX = x
                if (x > maxX) maxX = x
            }
            (maxX - minX).coerceIn(0.10f, 0.95f)
        }

        val smallFaceT = ((0.34f - faceWidthUv) / 0.18f).coerceIn(0f, 1f) // 0=normal/large, 1=small
        val faceBoost = (1.0f + 0.14f * smallFaceT).coerceIn(1.0f, 1.18f)

        // Low lighting heuristic (dimmer face mean luma): soften overall contour so it doesn't look harsh.
        val lowLightT = ((0.42f - lighting.faceMeanY) / 0.22f).coerceIn(0f, 1f)
        val lowLightSoft = (1.0f - 0.18f * lowLightT).coerceIn(0.78f, 1.0f)

        // Head pose from landmarks (pitch/yaw/roll).
        val pose = ContourLandmarkMapping.estimateHeadPoseMediaPipe478(lm)
        val lowAngleT = pose.lowAngleT
        val highAngleT = pose.highAngleT
        val yawT = pose.yawT
        val noseCenterT = run {
            val x = lm.getOrNull(com.ham.app.data.LandmarkIndex.NOSE_TIP * 3)?.coerceIn(0f, 1f) ?: 0.5f
            (1.0f - kotlin.math.abs(x - 0.5f) / 0.22f).coerceIn(0f, 1f)
        }

        // Region multipliers:
        // Keep these in the "reads as sculpt, not paint" range.
        val cheekBase = cheekMultiplier.coerceIn(0f, 1f)
        val jawBase = jawMultiplier.coerceIn(0f, 1f)
        // Temples / hairline.
        val foreheadBase = templeMultiplier.coerceIn(0f, 1f)
        val noseBase = noseMultiplier.coerceIn(0f, 1f)

        val cheekPose = (1.0f + 0.20f * lowAngleT + 0.12f * highAngleT).coerceIn(0.75f, 1.45f)
        val jawPose = (1.0f + 0.30f * lowAngleT - 0.35f * highAngleT).coerceIn(0.55f, 1.55f)
        val foreheadPose = (1.0f + 0.10f * lowAngleT + 0.08f * highAngleT).coerceIn(0.70f, 1.25f)
        // Nose contour must fade in profile to avoid artifacts.
        val noseYawMul = (1.0f - 0.85f * yawT).coerceAtLeast(0.10f)
        val nosePose =
            (0.55f + 0.45f * noseCenterT).coerceIn(0.55f, 1.0f) *
                noseYawMul

        // Map style alphas into the relight shader's base strengths.
        // These are intentionally stronger than the raw style alpha so the result reads on-device,
        // while the relight shader itself suppresses harshness in already-shadowed areas.
        val baseContour =
            (style.contourAlpha.coerceIn(0f, 1f) *
                contourIntensity.coerceIn(0f, 1f) *
                7.4f * faceBoost * lowLightSoft).coerceIn(0f, 1.0f)
        val baseHighlight =
            (style.highlightAlpha.coerceIn(0f, 1f) * 4.2f * faceBoost * lowLightSoft).coerceIn(0f, 1.0f)
        // Micro-contrast + spec are subtle “pro finish” touches, strongest when highlight is enabled.
        val baseMicro = (baseContour * 0.22f).coerceIn(0f, 0.35f)
        val baseSpec = (baseHighlight * 0.20f).coerceIn(0f, 0.35f)

        val cheek0 = (cheekBase * cheekPose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f)
        val jaw0 = (jawBase * jawPose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f)
        val nose0 = (noseBase * nosePose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f)
        val forehead0 = (foreheadBase * foreheadPose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f)

        // EMA (stable per-region intensity).
        val emaA = 0.10f
        if (!hasStrengthEma) {
            cheekEma = cheek0
            jawEma = jaw0
            noseEma = nose0
            foreheadEma = forehead0
            baseContourEma = baseContour
            hasStrengthEma = true
        } else {
            cheekEma += (cheek0 - cheekEma) * emaA
            jawEma += (jaw0 - jawEma) * emaA
            noseEma += (nose0 - noseEma) * emaA
            foreheadEma += (forehead0 - foreheadEma) * emaA
            baseContourEma += (baseContour - baseContourEma) * emaA
        }

        return Strengths(
            cheek = cheekEma.coerceIn(0f, 1.0f),
            jaw = jawEma.coerceIn(0f, 1.0f),
            nose = noseEma.coerceIn(0f, 1.0f),
            forehead = foreheadEma.coerceIn(0f, 1.0f),
            baseContour = baseContourEma.coerceIn(0f, 1.0f),
            baseHighlight = baseHighlight,
            baseMicro = baseMicro,
            baseSpec = baseSpec,
        )
    }

    /**
     * Computes contour shade from sampled skin tone per requirements:
     * contourColor = skinColor * 0.7, with a slightly reduced red channel.
     */
    fun computeContourShade(skin: Color): Pair<Color, Float> {
        // Cool-toned, slightly desaturated shadow tint:
        // - avoids orange/brown "dirt" read
        // - stays conservative to prevent muddy skin, especially in low light
        fun luma(c: Color) = c.red * 0.2126f + c.green * 0.7152f + c.blue * 0.0722f
        fun mix(a: Float, b: Float, t0: Float): Float {
            val t = t0.coerceIn(0f, 1f)
            return a + (b - a) * t
        }

        val y = luma(skin).coerceIn(0f, 1f)
        val desatT = 0.28f
        val sdR = mix(skin.red, y, desatT)
        val sdG = mix(skin.green, y, desatT)
        val sdB = mix(skin.blue, y, desatT)

        // Factor color (multiplied into the already-smoothed base):
        // keep it closer to 1 than previous (less muddy), and cool it slightly.
        val k = 0.74f
        val shade = Color(
            red = (sdR * k * 0.90f).coerceIn(0.10f, 0.98f),
            green = (sdG * k * 0.99f).coerceIn(0.10f, 0.98f),
            blue = (sdB * k * 1.04f).coerceIn(0.10f, 0.98f),
            alpha = 1f,
        )

        // Shader uses this very lightly; keep moderate.
        return Pair(shade, 0.66f)
    }

    fun renderToScreen(
        lm: FloatArray,
        isMirrored: Boolean,
        style: com.ham.app.data.MakeupStyle,
        skinBase: Color,
        frameTexture: Int,
        frameTexelSizeX: Float,
        frameTexelSizeY: Float,
        lighting: LightingMetrics,
        nowNs: Long,
    ): Boolean {
        if (!effect.isReady()) return false
        if (style.contourAlpha <= 0.01f && style.highlightAlpha <= 0.01f) return false

        val (shade, coolTone) = computeContourShade(skinBase)
        val strengths = computeStrengths(lm, style, lighting)

        val params = ContourParams(
            shade = shade,
            coolTone = coolTone,
            // Keep the packed mask normalized; final intensity is controlled by the relight shader bases.
            intensity = 1.0f,
            cheekContour = strengths.cheek,
            jawContour = strengths.jaw,
            noseContour = strengths.nose,
            foreheadContour = strengths.forehead,
            softness = 1.0f,
            faceErosionScale = 0.01f,
            // Soft-light reads more like “pro” sculpt (TikTok/Snap) than pure multiply.
            // Strengths already adapt to lighting/pose, so we keep those as-is.
            blendMode = ContourBlendMode.SOFT_LIGHT,
            landmarkSmoothingT = 0.20f,
            temporalSmoothing = 0.60f,
        )

        val maskTex = effect.buildContourMask(
            landmarks = lm,
            faceRect = null,
            params = params,
            isMirrored = isMirrored,
            nowNs = nowNs,
        )
        if (maskTex == 0) return false

        effect.renderContour(
            frameTexture = frameTexture,
            maskTexture = maskTex,
            params = params,
            baseContour = strengths.baseContour,
            baseHighlight = strengths.baseHighlight,
            baseMicro = strengths.baseMicro,
            baseSpec = strengths.baseSpec,
            faceMeanY = lighting.faceMeanY,
            faceStdY = lighting.faceStdY,
            clipFrac = lighting.clipFrac,
            lightBias = lighting.lightBias,
            frameTexelSizeX = frameTexelSizeX,
            frameTexelSizeY = frameTexelSizeY,
        )

        return true
    }

    /**
     * Build and return a contour mask for this frame (GPU).
     *
     * The returned [BuiltContour.maskTex] can be reused by:
     * - the multiply contour composite pass
     * - mask-aware smoothing in the foundation shader
     */
    fun buildMask(
        lm: FloatArray,
        isMirrored: Boolean,
        style: com.ham.app.data.MakeupStyle,
        skinBase: Color,
        lighting: LightingMetrics,
        nowNs: Long,
    ): BuiltContour? {
        if (!effect.isReady()) return null
        if (style.contourAlpha <= 0.01f) return null

        val (shade, coolTone) = computeContourShade(skinBase)
        val strengths = computeStrengths(lm, style, lighting)

        val params = ContourParams(
            shade = shade,
            coolTone = coolTone,
            intensity = 1.0f,
            cheekContour = strengths.cheek,
            jawContour = strengths.jaw,
            noseContour = strengths.nose,
            foreheadContour = strengths.forehead,
            softness = 1.0f,
            faceErosionScale = 0.01f,
            blendMode = ContourBlendMode.SOFT_LIGHT,
            landmarkSmoothingT = 0.20f,
            temporalSmoothing = 0.60f,
        )

        val maskTex = effect.buildContourMask(
            landmarks = lm,
            faceRect = null,
            params = params,
            isMirrored = isMirrored,
            nowNs = nowNs,
        )
        if (maskTex == 0) return null

        return BuiltContour(
            maskTex = maskTex,
            shade = shade,
            baseContour = strengths.baseContour,
        )
    }

    fun applyContourMultiply(
        built: BuiltContour,
        contrastBoost: Float,
    ) {
        effect.applyContourMultiply(
            maskTexture = built.maskTex,
            shade = built.shade,
            baseContour = built.baseContour,
            contrastBoost = contrastBoost,
        )
    }
}

