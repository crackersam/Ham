precision highp float;

uniform sampler2D uNewTex;
uniform sampler2D uPrevTex;
uniform float uT; // 0..1 (1 = all new)
uniform mat3 uUvTransform;

varying vec2 vTexCoord;

void main() {
    vec2 uv = (uUvTransform * vec3(vTexCoord, 1.0)).xy;
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    vec4 n = texture2D(uNewTex, uv);
    vec4 p = texture2D(uPrevTex, uv);
    gl_FragColor = mix(p, n, clamp(uT, 0.0, 1.0));
}

