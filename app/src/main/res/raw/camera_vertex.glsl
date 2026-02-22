attribute vec4 aPosition;
attribute vec2 aTexCoord;

// 1.0 for mirror preview, 0.0 for unmirrored
uniform float uMirror;
uniform vec2 uCropScale;

varying vec2 vTexCoord;

void main() {
    vec2 tc = aTexCoord;
    if (uMirror > 0.5) tc.x = 1.0 - tc.x;
    vTexCoord = tc;

    vec2 pos = aPosition.xy * uCropScale;
    gl_Position = vec4(pos, aPosition.z, aPosition.w);
}
