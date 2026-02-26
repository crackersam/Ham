attribute vec4 aPosition;
attribute vec2 aTexCoord;

uniform vec2 uCropScale;

varying vec2 vTexCoord;
varying vec2 vNdcPos;

void main() {
    vTexCoord = aTexCoord;
    // For the relight composite we always draw a full-screen quad (gl_Position uses raw aPosition),
    // but we still need "content NDC" (post-cropScale) for geometric masks (beard exclusion, etc).
    vNdcPos = aPosition.xy * uCropScale;
    gl_Position = vec4(aPosition.xy, aPosition.z, aPosition.w);
}

