precision highp float;

varying float vEdgeFactor;

void main() {
    // Match the feathering curve used by other makeup/foundation passes:
    // fully opaque interior, soft falloff only near the silhouette.
    float a = smoothstep(0.0, 0.20, vEdgeFactor)
            * smoothstep(0.0, 0.55, vEdgeFactor);
    gl_FragColor = vec4(a, a, a, a);
}

