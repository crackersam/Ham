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
        val master: Float,
        val cheek: Float,
        val jaw: Float,
        val nose: Float,
        val chin: Float,
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

        // Requested visible intensities (TikTok-level) as the base.
        // Pose adjusts these:
        // - low angle: cheek higher/stronger, jaw + chin stronger
        // - high angle: jaw softer, cheek emphasized
        val cheekBase = 0.60f
        val jawBase = 0.50f
        val chinBase = 0.50f
        val noseBase = 0.40f

        val cheekPose = (1.0f + 0.20f * lowAngleT + 0.12f * highAngleT).coerceIn(0.75f, 1.45f)
        val jawPose = (1.0f + 0.30f * lowAngleT - 0.35f * highAngleT).coerceIn(0.55f, 1.55f)
        val chinPose = (1.0f + 0.55f * lowAngleT - 0.25f * highAngleT).coerceIn(0.55f, 1.80f)
        val nosePose = (0.55f + 0.45f * noseCenterT).coerceIn(0.55f, 1.0f) * (1.0f - 0.70f * yawT).coerceIn(0.25f, 1.0f)

        return Strengths(
            // Style master. Most styles use small contourAlpha; upscale to reach visible sculpting.
            master = (style.contourAlpha.coerceIn(0f, 1f) * 5.0f * faceBoost * lowLightSoft).coerceIn(0f, 1.0f),
            cheek = (cheekBase * cheekPose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f),
            jaw = (jawBase * jawPose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f),
            nose = (noseBase * nosePose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f),
            chin = (chinBase * chinPose * faceBoost * lowLightSoft).coerceIn(0f, 1.0f),
        )
    }

    /**
     * Computes contour shade from sampled skin tone per requirements:
     * contourColor = skinColor * 0.7, with a slightly reduced red channel.
     */
    fun computeContourShade(skin: Color): Pair<Color, Float> {
        val k = 0.70f
        val shade = Color(
            red = (skin.red * k * 0.94f).coerceIn(0.02f, 0.98f),
            green = (skin.green * k).coerceIn(0.02f, 0.98f),
            blue = (skin.blue * k).coerceIn(0.02f, 0.98f),
            alpha = 1f,
        )
        // Keep a small coolTone bias for cross-device stability (used very lightly in shader).
        return Pair(shade, 0.65f)
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
        if (style.contourAlpha <= 0.01f) return false

        val (shade, coolTone) = computeContourShade(skinBase)
        val strengths = computeStrengths(lm, style, lighting)

        val params = ContourParams(
            shade = shade,
            coolTone = coolTone,
            intensity = strengths.master.coerceIn(0f, 1f),
            cheekContour = strengths.cheek,
            jawContour = strengths.jaw,
            noseContour = strengths.nose,
            chinContour = strengths.chin,
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
            baseContour = 1f,
            baseHighlight = 0f,
            baseMicro = 0f,
            baseSpec = 0f,
            faceMeanY = lighting.faceMeanY,
            faceStdY = lighting.faceStdY,
            clipFrac = lighting.clipFrac,
            lightBias = lighting.lightBias,
            frameTexelSizeX = frameTexelSizeX,
            frameTexelSizeY = frameTexelSizeY,
        )

        return true
    }
}

