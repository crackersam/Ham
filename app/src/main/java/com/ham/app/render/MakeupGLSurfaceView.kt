package com.ham.app.render

import android.content.Context
import android.graphics.SurfaceTexture
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import android.view.Surface
import androidx.camera.core.Preview
import androidx.core.content.ContextCompat

/**
 * GLSurfaceView that owns the camera OES texture and the makeup GL renderer.
 *
 * Key design decisions:
 *  - No alpha EGL config – we render opaque and don't need it; requesting
 *    alpha (8,8,8,8) causes EGLConfigChooser to throw on many devices.
 *  - Uses ContextCompat.getMainExecutor() instead of context.mainExecutor
 *    (the latter is API 28+, our minSdk is 26).
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

    /**
     * Returns a [Preview.SurfaceProvider] backed by the GL SurfaceTexture.
     * Must only be called after [renderer.surfaceTexture] is non-null (i.e.,
     * after [onSurfaceCreated] has fired on the GL thread).
     */
    fun buildSurfaceProvider(): Preview.SurfaceProvider = Preview.SurfaceProvider { request ->
        val st = renderer.surfaceTexture ?: run {
            request.willNotProvideSurface()
            return@SurfaceProvider
        }
        val resolution = request.resolution
        st.setDefaultBufferSize(resolution.width, resolution.height)
        val surface = Surface(st)
        request.provideSurface(surface, ContextCompat.getMainExecutor(context)) {
            surface.release()
        }
    }
}
