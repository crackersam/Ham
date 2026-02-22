package com.ham.app.render

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.AttributeSet

/**
 * GLSurfaceView that owns the makeup GL renderer.
 *
 * Key design decisions:
 *  - No alpha EGL config – we render opaque and don't need it; requesting
 *    alpha (8,8,8,8) causes EGLConfigChooser to throw on many devices.
 */
class MakeupGLSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : GLSurfaceView(context, attrs) {

    val renderer = MakeupGLRenderer(context)

    init {
        setEGLContextClientVersion(2)
        // Do NOT call setEGLConfigChooser – let GLSurfaceView pick the default
        // RGB(X) config.  Requesting an alpha channel (8,8,8,8) is unreliable
        // across devices and is unnecessary since we render opaque.
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY
    }
}
