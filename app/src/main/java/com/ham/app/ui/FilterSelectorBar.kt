package com.ham.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ham.app.data.MakeupStyle
import com.ham.app.data.MAKEUP_STYLES

@Composable
fun FilterSelectorBar(
    selectedIndex: Int,
    onStyleSelected: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    val listState = rememberLazyListState()

    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    listOf(Color.Transparent, Color(0xCC000000))
                )
            )
            .padding(bottom = 24.dp, top = 12.dp),
    ) {
        LazyRow(
            state = listState,
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            contentPadding = PaddingValues(horizontal = 24.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            itemsIndexed(MAKEUP_STYLES) { index, style ->
                FilterChip(
                    style = style,
                    isSelected = index == selectedIndex,
                    onClick = { onStyleSelected(index) },
                )
            }
        }
    }
}

@Composable
private fun FilterChip(
    style: MakeupStyle,
    isSelected: Boolean,
    onClick: () -> Unit,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .width(64.dp)
            .clickable(onClick = onClick),
    ) {
        // Colour swatch
        Box(
            modifier = Modifier
                .size(52.dp)
                .clip(CircleShape)
                .then(
                    if (isSelected) Modifier.border(3.dp, Color.White, CircleShape)
                    else Modifier.border(1.5.dp, Color(0x66FFFFFF), CircleShape)
                )
                .background(
                    brush = if (style.id == "none") {
                        Brush.radialGradient(
                            listOf(Color(0xFF444444), Color(0xFF222222))
                        )
                    } else {
                        Brush.sweepGradient(
                            listOf(
                                style.lipColor.copy(alpha = 1f),
                                style.eyeshadowColor.copy(alpha = 1f),
                                style.blushColor.copy(alpha = 1f),
                                style.lipColor.copy(alpha = 1f),
                            )
                        )
                    }
                ),
        ) {
            if (style.id == "none") {
                Text(
                    text = "✕",
                    color = Color.White,
                    fontSize = 20.sp,
                    modifier = Modifier.align(Alignment.Center),
                )
            }
        }

        Spacer(Modifier.height(6.dp))

        Text(
            text = style.name,
            color = if (isSelected) Color.White else Color(0xCCFFFFFF),
            fontSize = 11.sp,
            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
            textAlign = TextAlign.Center,
            maxLines = 2,
            lineHeight = 13.sp,
        )
    }
}
