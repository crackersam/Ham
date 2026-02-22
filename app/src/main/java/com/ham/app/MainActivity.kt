package com.ham.app

import android.Manifest
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import com.ham.app.ui.CameraScreen
import com.ham.app.ui.theme.HamTheme

@OptIn(ExperimentalPermissionsApi::class)
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Keep screen on while the app is open
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val app = application as HamApplication

        setContent {
            HamTheme {
                var modelReady by remember { mutableStateOf(app.modelReady) }

                // Listen for model download completion
                DisposableEffect(Unit) {
                    app.onModelReady = { modelReady = true }
                    onDispose { app.onModelReady = null }
                }

                val permissions = rememberMultiplePermissionsState(
                    listOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.RECORD_AUDIO,
                    )
                )

                when {
                    permissions.allPermissionsGranted -> {
                        CameraScreen(modelReady = modelReady)
                    }
                    permissions.shouldShowRationale -> {
                        PermissionRationale(
                            onGrant = { permissions.launchMultiplePermissionRequest() }
                        )
                    }
                    else -> {
                        LaunchedEffect(Unit) {
                            permissions.launchMultiplePermissionRequest()
                        }
                        // Show blank while awaiting
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Black),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PermissionRationale(onGrant: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp),
        ) {
            Text(
                "Ham needs Camera and Microphone access to apply\nmakeup filters and record videos.",
                color = Color.White,
                fontSize = 16.sp,
                textAlign = TextAlign.Center,
                lineHeight = 24.sp,
            )
            Spacer(Modifier.height(24.dp))
            Button(onClick = onGrant) {
                Text("Grant Access")
            }
        }
    }
}
