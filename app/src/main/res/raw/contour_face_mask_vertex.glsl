// Face clip mask vertex shader (GLES2)
attribute vec2 aPosition;    // unscaled NDC position (-1..1)
attribute float aEdgeFactor; // 0 at boundary, 1 at centroid

uniform vec2 uCropScale;

varying float vEdgeFactor;

void main() {
    vEdgeFactor = aEdgeFactor;
    vec2 ndcPos = aPosition * uCropScale;
    gl_Position = vec4(ndcPos, 0.0, 1.0);
}

