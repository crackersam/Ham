attribute vec4 aPosition;
attribute vec2 aTexCoord;

// Texture transform matrix from SurfaceTexture.getTransformMatrix()
uniform mat4 uTexTransform;

// 1.0 for mirror preview, -1.0 for unmirrored (photo/video save path)
uniform float uMirror;

varying vec2 vTexCoord;

void main() {
    // Apply the OES texture transform (handles rotation from camera sensor)
    vec4 tc = uTexTransform * vec4(aTexCoord, 0.0, 1.0);

    // Mirror horizontally for selfie-style preview
    // uMirror = 1.0  → mirror (front-camera selfie preview)
    // uMirror = -1.0 → no mirror (when reading pixels for save)
    float mirroredX = mix(tc.x, 1.0 - tc.x, max(0.0, uMirror));
    vTexCoord = vec2(mirroredX, tc.y);

    gl_Position = aPosition;
}
