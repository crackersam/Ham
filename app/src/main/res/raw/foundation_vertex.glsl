// NDC position of this face-oval vertex
attribute vec2  aPosition;

// 0.0 = face boundary, 1.0 = face centroid — drives feathering at oval edge
attribute float aEdgeFactor;

// Unused by foundation but kept so the same VBO layout works for all shaders
attribute vec2  aRegionUV;

uniform mat4  uTexTransform;
uniform float uMirror;

varying float vEdgeFactor;
varying vec2  vCamUV;

void main() {
    vEdgeFactor = aEdgeFactor;

    // Convert NDC position → screen UV [0,1] matching the camera convention:
    //   camera UV Y=0 is at the TOP of the image (Android buffer origin),
    //   NDC Y=+1 is the TOP of the screen, so:
    //       screen_u = (ndcX + 1) / 2
    //       screen_v = (1 - ndcY) / 2
    vec2 screenUV = vec2((aPosition.x + 1.0) * 0.5,
                         (1.0 - aPosition.y) * 0.5);

    // Apply the SurfaceTexture transform matrix so the sample lands on the
    // correct texel in the OES camera texture
    vCamUV = (uTexTransform * vec4(screenUV, 0.0, 1.0)).xy;

    // Mirror must match the camera background pass (camera_vertex.glsl).
    // Without this, the foundation samples the un-mirrored OES texel while
    // the camera draw shows the mirrored one, producing a ghostly overlay.
    if (uMirror > 0.5) vCamUV.x = 1.0 - vCamUV.x;

    gl_Position = vec4(aPosition, 0.0, 1.0);
}
