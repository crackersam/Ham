package com.ham.app.render

import android.content.Context
import android.opengl.GLES20
import android.util.Log

private const val TAG = "ShaderUtils"

fun loadRawString(context: Context, resId: Int): String =
    context.resources.openRawResource(resId).bufferedReader().readText()

fun compileShader(type: Int, source: String): Int {
    val shader = GLES20.glCreateShader(type)
    check(shader != 0) { "glCreateShader failed" }
    GLES20.glShaderSource(shader, source)
    GLES20.glCompileShader(shader)
    val status = IntArray(1)
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
    if (status[0] == 0) {
        val log = GLES20.glGetShaderInfoLog(shader)
        GLES20.glDeleteShader(shader)
        error("Shader compile error: $log")
    }
    return shader
}

fun linkProgram(vertSrc: String, fragSrc: String): Int {
    val vert = compileShader(GLES20.GL_VERTEX_SHADER, vertSrc)
    val frag = compileShader(GLES20.GL_FRAGMENT_SHADER, fragSrc)
    val prog = GLES20.glCreateProgram()
    check(prog != 0) { "glCreateProgram failed" }
    GLES20.glAttachShader(prog, vert)
    GLES20.glAttachShader(prog, frag)
    GLES20.glLinkProgram(prog)
    GLES20.glDeleteShader(vert)
    GLES20.glDeleteShader(frag)
    val status = IntArray(1)
    GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, status, 0)
    if (status[0] == 0) {
        val log = GLES20.glGetProgramInfoLog(prog)
        GLES20.glDeleteProgram(prog)
        error("Program link error: $log")
    }
    return prog
}

fun checkGlError(op: String) {
    val err = GLES20.glGetError()
    if (err != GLES20.GL_NO_ERROR) Log.w(TAG, "$op: glError 0x${Integer.toHexString(err)}")
}
