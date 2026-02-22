package com.ham.app.data

import androidx.compose.ui.graphics.Color

data class MakeupStyle(
    val id: String,
    val name: String,
    val lipColor: Color,
    val eyeshadowColor: Color,
    val blushColor: Color,
    val linerColor: Color,
    val lipAlpha: Float,
    val eyeshadowAlpha: Float,
    val blushAlpha: Float,
    val linerAlpha: Float,
    val highlightColor: Color = Color(0xFFFFF0E0),
    val highlightAlpha: Float = 0f,
    val foundationAlpha: Float = 0f,
    val browColor: Color = Color.Transparent,
    val browFillAlpha: Float = 0f,
    val browAlpha: Float = 0f,
    val lashColor: Color = Color.Transparent,
    val lashAlpha: Float = 0f,
)

val MAKEUP_STYLES = listOf(
    MakeupStyle(
        id = "none",
        name = "None",
        lipColor = Color.Transparent,
        eyeshadowColor = Color.Transparent,
        blushColor = Color.Transparent,
        linerColor = Color.Transparent,
        lipAlpha = 0f,
        eyeshadowAlpha = 0f,
        blushAlpha = 0f,
        linerAlpha = 0f,
    ),
    MakeupStyle(
        id = "soft-day",
        name = "Soft Day",
        lipColor = Color(0xFFb86e7b),
        eyeshadowColor = Color(0xFFc9b0bf),
        blushColor = Color(0xFFdfa0a4),
        linerColor = Color(0xFF3f3234),
        lipAlpha = 0.60f,
        eyeshadowAlpha = 0.35f,
        blushAlpha = 0.35f,
        linerAlpha = 0.55f,
        foundationAlpha = 0.25f,
        browColor = Color(0xFF3f3234),
        browFillAlpha = 0.22f,
        browAlpha = 0.14f,
        lashColor = Color(0xFF1a1418),
        lashAlpha = 0.38f,
    ),
    MakeupStyle(
        id = "classic-evening",
        name = "Classic Evening",
        lipColor = Color(0xFF9d314b),
        eyeshadowColor = Color(0xFF7a5a8c),
        blushColor = Color(0xFFc77786),
        linerColor = Color(0xFF23181f),
        lipAlpha = 0.78f,
        eyeshadowAlpha = 0.50f,
        blushAlpha = 0.45f,
        linerAlpha = 0.72f,
        foundationAlpha = 0.28f,
        browColor = Color(0xFF23181f),
        browFillAlpha = 0.26f,
        browAlpha = 0.18f,
        lashColor = Color(0xFF1a1418),
        lashAlpha = 0.48f,
    ),
    MakeupStyle(
        id = "bridal-glow",
        name = "Bridal Glow",
        lipColor = Color(0xFFcd8a86),
        eyeshadowColor = Color(0xFFd2b09a),
        blushColor = Color(0xFFe6aeb0),
        linerColor = Color(0xFF3d3034),
        lipAlpha = 0.70f,
        eyeshadowAlpha = 0.40f,
        blushAlpha = 0.42f,
        linerAlpha = 0.60f,
        highlightColor = Color(0xFFFFF8F0),
        highlightAlpha = 0.25f,
        foundationAlpha = 0.35f,
        browColor = Color(0xFF3d3034),
        browFillAlpha = 0.26f,
        browAlpha = 0.16f,
        lashColor = Color(0xFF1a1418),
        lashAlpha = 0.44f,
    ),
    MakeupStyle(
        id = "editorial",
        name = "Editorial",
        lipColor = Color(0xFFa82e57),
        eyeshadowColor = Color(0xFF8f4a87),
        blushColor = Color(0xFFd56582),
        linerColor = Color(0xFF1a1418),
        lipAlpha = 0.88f,
        eyeshadowAlpha = 0.60f,
        blushAlpha = 0.52f,
        linerAlpha = 0.85f,
        foundationAlpha = 0.28f,
        browColor = Color(0xFF1a1418),
        browFillAlpha = 0.30f,
        browAlpha = 0.22f,
        lashColor = Color(0xFF1a1418),
        lashAlpha = 0.56f,
    ),
)

// ─── MediaPipe FaceLandmarker 478-point landmark indices ───────────────────

/**
 * Face silhouette oval — 36 points, clockwise from forehead centre.
 * Used as the foundation mesh boundary so skin-smoothing and colour
 * correction are confined precisely to the face.
 */
val FACE_OVAL = intArrayOf(
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109
)

/** Outer lip ring – clockwise */
val LIPS_OUTER = intArrayOf(
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
)

/** Inner lip ring (mouth opening) – clockwise */
val LIPS_INNER = intArrayOf(
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
)

/** Cupid's bow highlight points */
val LIP_CUPID_BOW = intArrayOf(0, 37, 267)

/** Left eye socket / eyeshadow region */
val LEFT_EYE_SOCKET = intArrayOf(
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
)

/** Right eye socket / eyeshadow region */
val RIGHT_EYE_SOCKET = intArrayOf(
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466
)

/**
 * Left upper eyelid arc — inner corner → upper lash line → outer corner.
 * Contains NO lower-lid points so the eyeshadow mesh built from this arc
 * sits entirely above the lash line and never covers the eyeball.
 */
val LEFT_UPPER_LID_ARC = intArrayOf(133, 173, 157, 158, 159, 160, 161, 246, 33)

/**
 * Right upper eyelid arc — inner corner → upper lash line → outer corner.
 * Mirror of LEFT_UPPER_LID_ARC for the right eye.
 */
val RIGHT_UPPER_LID_ARC = intArrayOf(362, 398, 384, 385, 386, 387, 388, 466, 263)

/** Left eyeliner path (lower lash line) */
val LEFT_LINER = intArrayOf(33, 160, 159, 158, 157, 173, 133)

/** Right eyeliner path (lower lash line) */
val RIGHT_LINER = intArrayOf(263, 387, 386, 385, 384, 398, 362)

/** Left upper-lash liner */
val LEFT_UPPER_LINER = intArrayOf(246, 161, 160, 159, 158, 157, 173)

/** Right upper-lash liner */
val RIGHT_UPPER_LINER = intArrayOf(466, 388, 387, 386, 385, 384, 398)

/** Left cheek blush region */
val LEFT_CHEEK = intArrayOf(50, 123, 116, 117, 118, 101, 205, 36, 187, 147)

/** Right cheek blush region */
val RIGHT_CHEEK = intArrayOf(280, 352, 346, 347, 348, 330, 425, 266, 411, 376)

/** Left eyebrow */
val LEFT_BROW = intArrayOf(46, 53, 52, 65, 55, 70, 63, 105, 66, 107)

/** Right eyebrow */
val RIGHT_BROW = intArrayOf(276, 283, 282, 295, 285, 300, 293, 334, 296, 336)

/** Nose bridge for highlight */
val NOSE_BRIDGE = intArrayOf(168, 6, 197, 195, 5)

/** Forehead highlight / contour anchor points */
val FOREHEAD_CENTER = intArrayOf(10, 151, 9, 8)

/** Key point indices for single-point lookups */
object LandmarkIndex {
    const val NOSE_TIP = 1
    const val CHIN = 152
    const val LEFT_EYE_OUTER = 33
    const val LEFT_EYE_INNER = 133
    const val LEFT_EYE_LOWER = 145
    const val RIGHT_EYE_OUTER = 263
    const val RIGHT_EYE_INNER = 362
    const val RIGHT_EYE_LOWER = 374
    const val LIP_TOP_CENTER = 0
    const val LIP_BOTTOM_CENTER = 17
    const val LIP_LEFT = 61
    const val LIP_RIGHT = 291
    const val LEFT_CHEEK_CENTER = 205
    const val RIGHT_CHEEK_CENTER = 425
}
