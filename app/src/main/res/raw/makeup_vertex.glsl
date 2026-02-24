// NDC position of this makeup vertex
attribute vec2 aPosition;

// 0.0 = hard edge, 1.0 = center of region – drives soft feathering
attribute float aEdgeFactor;

// UV within region for gradient effects (optional, use for eyeshadow gradient)
attribute vec2 aRegionUV;

varying float vEdgeFactor;
varying vec2  vRegionUV;
varying vec2  vNdcPos;

uniform vec2 uCropScale;

void main() {
    vEdgeFactor = aEdgeFactor;
    vRegionUV   = aRegionUV;
    vNdcPos = aPosition * uCropScale;
    gl_Position = vec4(vNdcPos, 0.0, 1.0);
}
