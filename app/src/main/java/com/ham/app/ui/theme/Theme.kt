package com.ham.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = Pink80,
    secondary = PinkGrey80,
    tertiary = Peach80,
    background = Color.Black,
    surface = Color.Black,
)

@Composable
fun HamTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content,
    )
}
