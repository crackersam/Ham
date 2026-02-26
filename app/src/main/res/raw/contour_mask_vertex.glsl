attribute vec4 aPosition;
attribute vec2 aTexCoord;

uniform vec2 uCropScale;

varying vec2 vTexCoord;
varying vec2 vNdcPos;

void main() {
    vTexCoord = aTexCoord;
    vNdcPos = aPosition.xy * uCropScale;
    gl_Position = vec4(vNdcPos, aPosition.z, aPosition.w);
}

