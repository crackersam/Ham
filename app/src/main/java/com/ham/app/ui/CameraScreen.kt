package com.ham.app.ui

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.ViewGroup
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.ham.app.camera.CameraManager
import com.ham.app.data.MAKEUP_STYLES
import com.ham.app.face.FaceLandmarkerHelper
import com.ham.app.render.MakeupGLRenderer
import com.ham.app.render.MakeupGLSurfaceView
import com.ham.app.render.VideoRecorder
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val TAG = "CameraScreen"

@Composable
fun CameraScreen(modelReady: Boolean) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    var selectedStyleIndex by remember { mutableIntStateOf(1) }
    var filtersVisible by remember { mutableStateOf(false) }
    var isRecording by remember { mutableStateOf(false) }
    var flashVisible by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf<String?>(null) }
    var recordingSeconds by remember { mutableIntStateOf(0) }

    // Hold refs for GL view, camera, recorder
    val glSurfaceViewRef = remember { mutableStateOf<MakeupGLSurfaceView?>(null) }
    val cameraManager = remember { CameraManager(context) }
    val videoRecorder = remember { VideoRecorder(context) }
    val app = context.applicationContext as com.ham.app.HamApplication
    val landmarkerHelper = remember {
        FaceLandmarkerHelper(
            context = context,
            modelPath = app.modelPath(),
            onResult = { /* renderer is driven by CameraManager frame packets */ },
            onError = { e -> Log.e(TAG, "FaceLandmarker error", e) },
        )
    }

    // Recording timer
    LaunchedEffect(isRecording) {
        if (isRecording) {
            recordingSeconds = 0
            while (isRecording) {
                delay(1000)
                recordingSeconds++
            }
        }
    }

    // Auto-dismiss status messages
    LaunchedEffect(statusText) {
        if (statusText != null) {
            delay(2500)
            statusText = null
        }
    }

    // Update style when selection changes
    LaunchedEffect(selectedStyleIndex) {
        glSurfaceViewRef.value?.renderer?.currentStyle = MAKEUP_STYLES[selectedStyleIndex]
    }

    // When the model download finishes AFTER the GL surface is already up,
    // the AndroidView factory lambda has already captured modelReady=false, so
    // setup() was never called.  This effect re-calls it whenever modelReady
    // flips to true – idempotent because FaceLandmarkerHelper.setup() is
    // guarded by an executor.
    LaunchedEffect(modelReady) {
        if (modelReady) {
            landmarkerHelper.setup()
        }
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> {
                    // Re-setup after coming back from background (close() resets the flag)
                    if (modelReady) landmarkerHelper.setup()
                }
                Lifecycle.Event.ON_PAUSE -> {
                    if (isRecording) videoRecorder.stop()
                    landmarkerHelper.close()
                }
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            cameraManager.unbind()
            landmarkerHelper.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── GL Surface View (never overlaps the bottom bar) ─────────────────
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
            ) {
                val targetAspect = 9f / 16f
                val screenAspect = (maxWidth / maxHeight)
                val windowModifier =
                    if (screenAspect > targetAspect) {
                        // Screen is relatively wide: fit by height (bars left/right).
                        Modifier.fillMaxHeight().aspectRatio(targetAspect)
                    } else {
                        // Screen is relatively tall/narrow: fit by width (bars top/bottom).
                        Modifier.fillMaxWidth().aspectRatio(targetAspect)
                    }

                Box(
                    modifier = windowModifier
                        .align(Alignment.TopCenter)
                        .clip(RoundedCornerShape(26.dp))
                        .background(Color.Black),
                ) {
                    AndroidView(
                        factory = { ctx ->
                            MakeupGLSurfaceView(ctx).also { sv ->
                                sv.layoutParams = ViewGroup.LayoutParams(
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                )
                                glSurfaceViewRef.value = sv

                                // Always show the full camera frame in the preview window (no crop/distort).
                                sv.renderer.previewScaleMode = MakeupGLRenderer.PreviewScaleMode.FIT_CENTER

                                // Set initial style
                                sv.renderer.currentStyle = MAKEUP_STYLES[selectedStyleIndex]

                                // CameraX binding must happen on the main thread.
                                Handler(Looper.getMainLooper()).post {
                                    if (modelReady) landmarkerHelper.setup()
                                    cameraManager.bind(lifecycleOwner, sv, landmarkerHelper)
                                }
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }

            // ── Bottom controls (separate from preview) ─────────────────────────
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(bottom = 8.dp),
            ) {

                // Filter selector
                AnimatedVisibility(
                    visible = filtersVisible,
                    enter = fadeIn(),
                    exit = fadeOut(),
                ) {
                    FilterSelectorBar(
                        selectedIndex = selectedStyleIndex,
                        onStyleSelected = { idx ->
                            selectedStyleIndex = idx
                            glSurfaceViewRef.value?.renderer?.currentStyle = MAKEUP_STYLES[idx]
                        },
                    )
                }

                // Capture / Record row
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xCC000000))
                        .padding(vertical = 20.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 32.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        // Photo shutter
                        ShutterButton(
                            label = "Photo",
                            isDestructive = false,
                            onClick = {
                                val sv = glSurfaceViewRef.value ?: return@ShutterButton
                                flashVisible = true
                                scope.launch { delay(80); flashVisible = false }
                                sv.renderer.onPixelsReady = { bytes, w, h ->
                                    savePhoto(context, bytes, w, h) { msg ->
                                        scope.launch { statusText = msg }
                                    }
                                    sv.renderer.onPixelsReady = null
                                }
                                sv.queueEvent { sv.renderer.requestCapture() }
                            },
                        )

                        // Record button
                        ShutterButton(
                            label = if (isRecording) "Stop" else "Video",
                            isDestructive = isRecording,
                            onClick = {
                                val sv = glSurfaceViewRef.value ?: return@ShutterButton
                                if (!isRecording) {
                                    isRecording = true
                                    videoRecorder.onRecordingFinished = { _ ->
                                        scope.launch { statusText = "Video saved!" }
                                    }
                                    videoRecorder.start(sv, sv.width, sv.height)
                                } else {
                                    isRecording = false
                                    videoRecorder.stop()
                                }
                            },
                        )

                        // Makeup styles toggle (shows/hides the selector bar)
                        ShutterButton(
                            label = "Makeup",
                            isDestructive = false,
                            onClick = { filtersVisible = !filtersVisible },
                        )
                    }
                }
            }
        }

        // ── Loading overlay (while model is downloading) ──────────────────────
        AnimatedVisibility(
            visible = !modelReady,
            enter = fadeIn(),
            exit = fadeOut(),
        ) {
            Box(
                modifier = Modifier.fillMaxSize().background(Color(0xCC000000)),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator(color = Color.White)
                    Spacer(Modifier.height(16.dp))
                    Text(
                        "Preparing Ham filters…",
                        color = Color.White,
                        fontSize = 16.sp,
                    )
                }
            }
        }

        // ── Recording timer badge ─────────────────────────────────────────────
        if (isRecording) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 52.dp)
                    .clip(CircleShape)
                    .background(Color(0xCCFF3B30))
                    .padding(horizontal = 14.dp, vertical = 6.dp),
            ) {
                PulsingDot()
                Spacer(Modifier.width(6.dp))
                Text(
                    text = formatSeconds(recordingSeconds),
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 15.sp,
                )
            }
        }

        // ── Status toast ──────────────────────────────────────────────────────
        AnimatedVisibility(
            visible = statusText != null,
            enter = fadeIn(),
            exit = fadeOut(),
            modifier = Modifier.align(Alignment.TopCenter).padding(top = 52.dp),
        ) {
            statusText?.let { msg ->
                Box(
                    modifier = Modifier
                        .clip(CircleShape)
                        .background(Color(0xCC000000))
                        .padding(horizontal = 20.dp, vertical = 10.dp),
                ) {
                    Text(msg, color = Color.White, fontSize = 14.sp)
                }
            }
        }

        // ── Flash overlay ─────────────────────────────────────────────────────
        AnimatedVisibility(
            visible = flashVisible,
            enter = fadeIn(tween(40)),
            exit = fadeOut(tween(180)),
        ) {
            Box(
                modifier = Modifier.fillMaxSize().background(Color.White),
            )
        }
    }
}

// ── Sub-composables ───────────────────────────────────────────────────────────

@Composable
private fun ShutterButton(
    label: String,
    isDestructive: Boolean,
    onClick: () -> Unit,
) {
    val scale by animateFloatAsState(
        targetValue = 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "scale",
    )

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            label,
            color = Color.White,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
        )
        Spacer(Modifier.height(6.dp))
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(68.dp)
                .scale(scale)
                .clip(CircleShape)
                .border(3.dp, Color.White, CircleShape)
                .background(
                    if (isDestructive) Color(0xCCFF3B30) else Color(0x33FFFFFF)
                )
                .clickable(onClick = onClick),
        ) {
            if (isDestructive) {
                Box(
                    modifier = Modifier
                        .size(22.dp)
                        .background(Color.White, shape = androidx.compose.foundation.shape.RoundedCornerShape(4.dp))
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(52.dp)
                        .clip(CircleShape)
                        .background(Color.White),
                )
            }
        }
    }
}

@Composable
private fun PulsingDot() {
    val inf = rememberInfiniteTransition(label = "pulse")
    val alpha by inf.animateFloat(
        initialValue = 1f, targetValue = 0.3f,
        animationSpec = infiniteRepeatable(
            animation = tween(600, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "alpha",
    )
    Box(
        modifier = Modifier
            .size(8.dp)
            .clip(CircleShape)
            .background(Color.White.copy(alpha = alpha)),
    )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

private fun formatSeconds(s: Int): String =
    "%02d:%02d".format(s / 60, s % 60)

private fun savePhoto(
    context: Context,
    rgba: ByteArray,
    width: Int,
    height: Int,
    onDone: (String) -> Unit,
) {
    try {
        // Convert RGBA bytes → Bitmap
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val buf = java.nio.ByteBuffer.wrap(rgba)
        bmp.copyPixelsFromBuffer(buf)

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "ham_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH,
                    "${Environment.DIRECTORY_PICTURES}/Ham")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri != null) {
            resolver.openOutputStream(uri)?.use { out ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
        }
        bmp.recycle()
        onDone("Photo saved!")
    } catch (e: Exception) {
        Log.e(TAG, "savePhoto error", e)
        onDone("Save failed")
    }
}
