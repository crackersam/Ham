/**
 * Professional Makeup Studio - Snapchat/Instagram Quality
 * 
 * Advanced Features Implemented:
 * - Kalman filtering for ultra-smooth face tracking
 * - 60fps inference for responsive tracking (16ms frame time)
 * - Adaptive smoothing based on motion detection
 * - Advanced bilateral filtering with skin detection
 * - HSV-based skin tone detection for selective processing
 * - Professional multi-pass rendering with proper blending
 * - Lighting-aware makeup application
 * - Realistic texture/grain addition for natural look
 * - Temporal smoothing for settings transitions
 * - Subsurface scattering simulation
 * - Performance optimization with dirty region tracking
 * - High-precision shader (highp float)
 * - 1024x1024 mask resolution for smooth edges
 * - Multi-pass blur for ultra-smooth feathering
 * - 1080p camera support with 60fps capability
 */

"use client";

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { useEffect, useMemo, useRef, useState } from "react";

type FacePoint = { x: number; y: number; z: number };
type CanvasPoint = { x: number; y: number };

type BeautyRenderer = {
  resize: (width: number, height: number) => void;
  render: (
    video: HTMLVideoElement,
    maskCanvas: HTMLCanvasElement,
    smoothAmount: number,
    toneAmount: number,
    sharpenAmount: number
  ) => void;
  dispose: () => void;
};

// Kalman filter for smooth landmark tracking (Snapchat/Instagram quality)
class KalmanFilter {
  private x: number;
  private p: number;
  private q: number;
  private r: number;
  private velocity: number;

  constructor(processNoise = 0.01, measurementNoise = 0.1) {
    this.x = 0;
    this.p = 1;
    this.q = processNoise;
    this.r = measurementNoise;
    this.velocity = 0;
  }

  update(measurement: number): number {
    // Prediction with velocity
    this.x = this.x + this.velocity;
    this.p = this.p + this.q;

    // Update
    const k = this.p / (this.p + this.r);
    const innovation = measurement - this.x;
    this.x = this.x + k * innovation;
    this.p = (1 - k) * this.p;
    
    // Update velocity
    this.velocity = this.velocity * 0.8 + innovation * 0.2;

    return this.x;
  }
}

// Kalman filter for each landmark point
class LandmarkKalmanFilter {
  private filtersX: KalmanFilter[];
  private filtersY: KalmanFilter[];
  private filtersZ: KalmanFilter[];

  constructor(numLandmarks: number) {
    this.filtersX = Array.from({ length: numLandmarks }, () => new KalmanFilter(0.008, 0.1));
    this.filtersY = Array.from({ length: numLandmarks }, () => new KalmanFilter(0.008, 0.1));
    this.filtersZ = Array.from({ length: numLandmarks }, () => new KalmanFilter(0.008, 0.15));
  }

  filter(points: FacePoint[]): FacePoint[] {
    return points.map((point, i) => ({
      x: this.filtersX[i].update(point.x),
      y: this.filtersY[i].update(point.y),
      z: this.filtersZ[i].update(point.z),
    }));
  }
}

// Motion detector for adaptive smoothing
class MotionDetector {
  private previousPoints: FacePoint[] | null = null;

  detectMotion(points: FacePoint[]): number {
    if (!this.previousPoints || this.previousPoints.length !== points.length) {
      this.previousPoints = points;
      return 0;
    }

    let totalMotion = 0;
    for (let i = 0; i < points.length; i++) {
      const dx = points[i].x - this.previousPoints[i].x;
      const dy = points[i].y - this.previousPoints[i].y;
      totalMotion += Math.sqrt(dx * dx + dy * dy);
    }

    this.previousPoints = points;
    return totalMotion / points.length;
  }
}

// Performance optimizer - skip rendering when no significant changes (Instagram quality)
class DirtyRegionTracker {
  private lastPoints: FacePoint[] | null = null;
  private lastStyle: string | null = null;
  private lastIntensity: number = 0;
  private framesSinceUpdate: number = 0;

  shouldRender(
    points: FacePoint[] | null,
    styleId: string,
    intensity: number,
    minMotionThreshold = 0.0008
  ): boolean {
    // Always render first frame or when no face
    if (!this.lastPoints || !points) {
      this.lastPoints = points;
      this.lastStyle = styleId;
      this.lastIntensity = intensity;
      this.framesSinceUpdate = 0;
      return true;
    }

    // Render if style or intensity changed significantly
    if (
      this.lastStyle !== styleId ||
      Math.abs(this.lastIntensity - intensity) > 0.01
    ) {
      this.lastPoints = points;
      this.lastStyle = styleId;
      this.lastIntensity = intensity;
      this.framesSinceUpdate = 0;
      return true;
    }

    // Calculate motion
    let motion = 0;
    for (let i = 0; i < Math.min(points.length, this.lastPoints.length); i++) {
      const dx = points[i].x - this.lastPoints[i].x;
      const dy = points[i].y - this.lastPoints[i].y;
      motion += Math.sqrt(dx * dx + dy * dy);
    }
    motion /= points.length;

    // Render if significant motion or every few frames
    this.framesSinceUpdate++;
    if (motion > minMotionThreshold || this.framesSinceUpdate > 5) {
      this.lastPoints = points;
      this.framesSinceUpdate = 0;
      return true;
    }

    return false;
  }
}

// Temporal value smoother for settings (prevents jarring changes)
class ValueSmoother {
  private current: number;
  private target: number;
  private speed: number;

  constructor(initial: number, speed = 0.15) {
    this.current = initial;
    this.target = initial;
    this.speed = speed;
  }

  setTarget(value: number) {
    this.target = value;
  }

  update(): number {
    this.current += (this.target - this.current) * this.speed;
    return this.current;
  }

  getCurrent(): number {
    return this.current;
  }
}

type MakeupStyle = {
  id: string;
  name: string;
  lipstick: string;
  eyeshadow: string;
  blush: string;
  liner: string;
  lipstickOpacity: number;
  eyeshadowOpacity: number;
  blushOpacity: number;
  linerOpacity: number;
};

const STYLES: MakeupStyle[] = [
  {
    id: "soft-day",
    name: "Soft Day",
    lipstick: "#c97888",
    eyeshadow: "#d4b5d8",
    blush: "#e8a5a8",
    liner: "#4d3d3f",
    lipstickOpacity: 0.68,
    eyeshadowOpacity: 0.50,
    blushOpacity: 0.48,
    linerOpacity: 0.82,
  },
  {
    id: "classic-evening",
    name: "Classic Evening",
    lipstick: "#a63852",
    eyeshadow: "#8e6b9f",
    blush: "#d88090",
    liner: "#2b1f26",
    lipstickOpacity: 0.85,
    eyeshadowOpacity: 0.65,
    blushOpacity: 0.60,
    linerOpacity: 0.98,
  },
  {
    id: "bridal-glow",
    name: "Bridal Glow",
    lipstick: "#d98e8d",
    eyeshadow: "#dbb5a0",
    blush: "#f0b5b0",
    liner: "#4a3a3e",
    lipstickOpacity: 0.75,
    eyeshadowOpacity: 0.52,
    blushOpacity: 0.58,
    linerOpacity: 0.85,
  },
  {
    id: "editorial",
    name: "Editorial",
    lipstick: "#b8325f",
    eyeshadow: "#9e5496",
    blush: "#de6f8a",
    liner: "#1a1418",
    lipstickOpacity: 0.95,
    eyeshadowOpacity: 0.72,
    blushOpacity: 0.68,
    linerOpacity: 1.0,
  },
];

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
const LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const LEFT_CHEEK = [50, 123, 116, 117, 118, 101, 205];
const RIGHT_CHEEK = [280, 352, 346, 347, 348, 330, 425];
const LEFT_LINER = [33, 160, 159, 158, 157, 173, 133];
const RIGHT_LINER = [263, 387, 386, 385, 384, 398, 362];
const LEFT_BROW = [70, 63, 105, 66, 107];
const RIGHT_BROW = [336, 296, 334, 293, 300];
const LEFT_EYE_SOCKET = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE_SOCKET = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const LEFT_EYE_OUTER = 33;
const LEFT_EYE_INNER = 133;
const LEFT_EYE_LOWER = 145;
const RIGHT_EYE_OUTER = 263;
const RIGHT_EYE_INNER = 362;
const RIGHT_EYE_LOWER = 374;
const LIP_HIGHLIGHT_ANCHOR = [0, 13];

function createShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? "Unknown shader error";
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vertex: string, fragment: string) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertex);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragment);
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create GL program");
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? "Unknown link error";
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return program;
}

function createBeautyRenderer(canvas: HTMLCanvasElement): BeautyRenderer {
  const gl = canvas.getContext("webgl", { 
    alpha: false, 
    antialias: true,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
    failIfMajorPerformanceCaveat: false
  });
  if (!gl) throw new Error("WebGL is unavailable in this browser/webview.");
  
  // Enable blending for iOS compatibility
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const vertexSource = `
    attribute vec2 aPosition;
    varying vec2 vUv;
    void main() {
      vUv = (aPosition + 1.0) * 0.5;
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }
  `;

  const fragmentSource = `
    precision highp float;
    varying vec2 vUv;

    uniform sampler2D uVideo;
    uniform sampler2D uMask;
    uniform vec2 uTexel;
    uniform float uSmooth;
    uniform float uTone;
    uniform float uSharpen;

    // Perlin-like noise for texture realism
    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    vec3 sampleVideo(vec2 uv) {
      return texture2D(uVideo, uv).rgb;
    }

    // Advanced skin detection using HSV
    bool isSkinTone(vec3 color) {
      float r = color.r;
      float g = color.g;
      float b = color.b;
      
      // RGB to HSV conversion
      float maxC = max(max(r, g), b);
      float minC = min(min(r, g), b);
      float delta = maxC - minC;
      
      float hue = 0.0;
      if (delta > 0.0) {
        if (maxC == r) {
          hue = mod((g - b) / delta, 6.0);
        } else if (maxC == g) {
          hue = (b - r) / delta + 2.0;
        } else {
          hue = (r - g) / delta + 4.0;
        }
        hue = hue * 60.0;
      }
      
      float saturation = maxC > 0.0 ? delta / maxC : 0.0;
      float value = maxC;
      
      // Skin tone ranges (tuned for diverse skin tones)
      bool hueRange = (hue >= 0.0 && hue <= 50.0) || (hue >= 340.0 && hue <= 360.0);
      bool satRange = saturation >= 0.20 && saturation <= 0.85;
      bool valRange = value >= 0.30 && value <= 1.0;
      
      return hueRange && satRange && valRange;
    }

    // Professional bilateral filter with depth awareness
    vec3 bilateralBlur(vec2 uv, vec3 center) {
      vec3 sum = vec3(0.0);
      float totalWeight = 0.0;
      
      // More samples for higher quality
      const int radius = 3;
      for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
          vec2 offset = vec2(float(x), float(y)) * uTexel * 1.5;
          vec3 sample = sampleVideo(uv + offset);
          
          // Spatial weight (Gaussian)
          float spatialDist = length(vec2(float(x), float(y)));
          float spatialWeight = exp(-spatialDist * spatialDist / 8.0);
          
          // Color weight (preserve edges)
          float colorDist = length(sample - center);
          float colorWeight = exp(-colorDist * colorDist / 0.15);
          
          // Skin detection weight (blur skin only)
          float skinWeight = isSkinTone(sample) ? 1.0 : 0.3;
          
          float weight = spatialWeight * colorWeight * skinWeight;
          sum += sample * weight;
          totalWeight += weight;
        }
      }
      
      return totalWeight > 0.0 ? sum / totalWeight : center;
    }

    void main() {
      // Mirror selfie feed and flip vertically for correct camera orientation
      vec2 uv = vec2(1.0 - vUv.x, 1.0 - vUv.y);
      vec3 center = sampleVideo(uv);
      float mask = texture2D(uMask, uv).r;
      
      // Detect if this is skin
      float skinMask = isSkinTone(center) ? 1.0 : 0.0;
      float effectiveMask = mask * skinMask;

      // Professional bilateral blur (only on skin)
      vec3 blurred = bilateralBlur(uv, center);
      
      // Edge detection for feature preservation
      vec3 edgeH = sampleVideo(uv + vec2(uTexel.x, 0.0)) - sampleVideo(uv - vec2(uTexel.x, 0.0));
      vec3 edgeV = sampleVideo(uv + vec2(0.0, uTexel.y)) - sampleVideo(uv - vec2(0.0, uTexel.y));
      float edge = length(edgeH) + length(edgeV);
      float edgeFactor = smoothstep(0.1, 0.3, edge);
      
      // Adaptive smoothing based on edges
      float smoothFactor = uSmooth * (1.0 - edgeFactor * 0.85) * effectiveMask;
      vec3 smoothed = mix(center, blurred, smoothFactor);

      // Unsharp masking for professional sharpening
      vec3 detail = smoothed - blurred;
      vec3 sharpened = smoothed + detail * (0.4 + uSharpen * 0.8) * (1.0 - effectiveMask * 0.5);

      // Convert to approximate LAB for better color handling
      float luma = dot(sharpened, vec3(0.2126, 0.7152, 0.0722));
      
      // Saturation boost (vibrant but natural)
      vec3 satBoosted = mix(vec3(luma), sharpened, 1.0 + uTone * 0.18);
      
      // Warm glow with lighting awareness (detect bright areas)
      float brightness = luma;
      vec3 warmTone = vec3(0.028, 0.016, 0.009) * uTone * smoothstep(0.3, 0.7, brightness);
      vec3 warmed = satBoosted + warmTone;
      
      // Skin brightening (professional technique)
      float skinBrightness = effectiveMask * uTone * 0.015;
      vec3 brightened = warmed + vec3(skinBrightness * 1.1, skinBrightness * 0.95, skinBrightness * 0.85);
      
      // Soft focus glow for that filter look
      vec3 softGlow = mix(brightened, blurred, 0.12 * uSmooth * effectiveMask);
      
      // Add subtle grain for realism (real makeup isn't perfectly smooth)
      float grain = (noise(uv * 800.0) - 0.5) * 0.008 * effectiveMask;
      vec3 textured = softGlow + vec3(grain);
      
      // Subsurface scattering simulation (red tint in shadows on skin)
      float shadowMask = smoothstep(0.6, 0.3, luma) * effectiveMask;
      vec3 subsurface = textured + vec3(0.012, 0.004, 0.002) * shadowMask * uTone;
      
      // Final blend with smooth transition
      vec3 beauty = clamp(subsurface, 0.0, 1.0);
      float blendMask = smoothstep(0.0, 0.2, mask) * smoothstep(1.0, 0.8, mask);
      vec3 outColor = mix(center, beauty, blendMask);
      
      gl_FragColor = vec4(outColor, 1.0);
    }
  `;

  const program = createProgram(gl, vertexSource, fragmentSource);
  gl.useProgram(program);

  const quad = gl.createBuffer();
  if (!quad) throw new Error("Failed to create vertex buffer");
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

  const aPosition = gl.getAttribLocation(program, "aPosition");
  gl.enableVertexAttribArray(aPosition);
  gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

  const uVideo = gl.getUniformLocation(program, "uVideo");
  const uMask = gl.getUniformLocation(program, "uMask");
  const uTexel = gl.getUniformLocation(program, "uTexel");
  const uSmooth = gl.getUniformLocation(program, "uSmooth");
  const uTone = gl.getUniformLocation(program, "uTone");
  const uSharpen = gl.getUniformLocation(program, "uSharpen");

  const videoTexture = gl.createTexture();
  const maskTexture = gl.createTexture();
  if (!videoTexture || !maskTexture) throw new Error("Failed to create texture resources");

  const setupTexture = (texture: WebGLTexture) => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  };
  setupTexture(videoTexture);
  setupTexture(maskTexture);

  let width = 0;
  let height = 0;

  const resize = (nextWidth: number, nextHeight: number) => {
    width = nextWidth;
    height = nextHeight;
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
      // iOS Safari: Clear any previous render state
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.viewport(0, 0, nextWidth, nextHeight);
  };

  const render = (
    video: HTMLVideoElement,
    maskCanvas: HTMLCanvasElement,
    smoothAmount: number,
    toneAmount: number,
    sharpenAmount: number
  ) => {
    if (!width || !height) return;
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    gl.uniform1i(uVideo, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, maskTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, gl.LUMINANCE, gl.UNSIGNED_BYTE, maskCanvas);
    gl.uniform1i(uMask, 1);

    gl.uniform2f(uTexel, 1 / width, 1 / height);
    gl.uniform1f(uSmooth, smoothAmount);
    gl.uniform1f(uTone, toneAmount);
    gl.uniform1f(uSharpen, sharpenAmount);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  const dispose = () => {
    gl.deleteTexture(videoTexture);
    gl.deleteTexture(maskTexture);
    gl.deleteBuffer(quad);
    gl.deleteProgram(program);
  };

  return { resize, render, dispose };
}

function hexToRgba(hex: string, alpha: number) {
  const cleaned = hex.replace("#", "");
  const chunk = cleaned.length === 3 ? cleaned.split("").map((v) => v + v).join("") : cleaned;
  const value = Number.parseInt(chunk, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Generate realistic texture/grain for makeup (Snapchat/Instagram quality)
function addMakeupTexture(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  intensity: number
) {
  if (intensity <= 0) return;
  
  const imageData = ctx.getImageData(x, y, width, height);
  const data = imageData.data;
  
  // Add subtle grain to simulate real makeup texture
  for (let i = 0; i < data.length; i += 4) {
    const px = (i / 4) % width;
    const py = Math.floor((i / 4) / width);
    
    // Simple pseudo-random noise based on position
    const noise = (Math.sin(px * 12.9898 + py * 78.233) * 43758.5453) % 1;
    const grain = (noise - 0.5) * intensity * 3;
    
    data[i] = Math.max(0, Math.min(255, data[i] + grain));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + grain));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + grain));
  }
  
  ctx.putImageData(imageData, x, y);
}

// Detect lighting/brightness in face region for adaptive makeup
function detectFaceLighting(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number
): number {
  // Sample center of face for lighting
  const noseTip = points[1]; // Nose tip is a good reference point
  if (!noseTip) return 0.5;
  
  const sampleX = Math.floor(noseTip.x * width);
  const sampleY = Math.floor(noseTip.y * height);
  const sampleSize = 20;
  
  try {
    const imageData = ctx.getImageData(
      Math.max(0, sampleX - sampleSize / 2),
      Math.max(0, sampleY - sampleSize / 2),
      sampleSize,
      sampleSize
    );
    
    const data = imageData.data;
    let totalBrightness = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      // Calculate luminance
      const luma = (data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722) / 255;
      totalBrightness += luma;
    }
    
    return totalBrightness / (data.length / 4);
  } catch {
    return 0.5;
  }
}

function tracePolygonPath(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  indices: number[],
  width: number,
  height: number
) {
  const first = points[indices[0]];
  if (!first) return false;
  ctx.beginPath();
  ctx.moveTo(first.x * width, first.y * height);
  for (let i = 1; i < indices.length; i += 1) {
    const p = points[indices[i]];
    if (!p) continue;
    ctx.lineTo(p.x * width, p.y * height);
  }
  ctx.closePath();
  return true;
}

function toCanvasPathPoints(points: FacePoint[], indices: number[], width: number, height: number): CanvasPoint[] {
  return indices
    .map((index) => points[index])
    .filter((point): point is FacePoint => Boolean(point))
    .map((point) => ({ x: point.x * width, y: point.y * height }));
}

function traceSmoothPath(ctx: CanvasRenderingContext2D, pathPoints: CanvasPoint[], closed: boolean) {
  if (pathPoints.length < 2) return false;
  if (pathPoints.length === 2) {
    ctx.beginPath();
    ctx.moveTo(pathPoints[0].x, pathPoints[0].y);
    ctx.lineTo(pathPoints[1].x, pathPoints[1].y);
    if (closed) ctx.closePath();
    return true;
  }

  ctx.beginPath();
  if (closed) {
    const last = pathPoints[pathPoints.length - 1];
    const first = pathPoints[0];
    const firstMid = { x: (last.x + first.x) * 0.5, y: (last.y + first.y) * 0.5 };
    ctx.moveTo(firstMid.x, firstMid.y);

    for (let i = 0; i < pathPoints.length; i += 1) {
      const current = pathPoints[i];
      const next = pathPoints[(i + 1) % pathPoints.length];
      const mid = { x: (current.x + next.x) * 0.5, y: (current.y + next.y) * 0.5 };
      ctx.quadraticCurveTo(current.x, current.y, mid.x, mid.y);
    }
    ctx.closePath();
    return true;
  }

  ctx.moveTo(pathPoints[0].x, pathPoints[0].y);
  for (let i = 1; i < pathPoints.length - 1; i += 1) {
    const current = pathPoints[i];
    const next = pathPoints[i + 1];
    const mid = { x: (current.x + next.x) * 0.5, y: (current.y + next.y) * 0.5 };
    ctx.quadraticCurveTo(current.x, current.y, mid.x, mid.y);
  }
  const last = pathPoints[pathPoints.length - 1];
  ctx.lineTo(last.x, last.y);
  return true;
}

// Adaptive smoothing based on motion (Instagram/Snapchat quality)
function smoothLandmarksAdaptive(
  previous: FacePoint[] | null,
  next: FacePoint[],
  motion: number
): FacePoint[] {
  if (!previous || previous.length !== next.length) return next;
  
  // Adaptive weight: faster smoothing when moving, slower when still
  // High motion (0.1+) = 0.6-0.7 weight (responsive)
  // Low motion (0.01) = 0.2-0.3 weight (stable)
  const baseWeight = 0.3;
  const motionFactor = Math.min(motion * 5.0, 1.0);
  const adaptiveWeight = baseWeight + motionFactor * 0.4;
  
  const w = Math.min(1, Math.max(0, adaptiveWeight));
  const prevW = 1 - w;
  
  return next.map((point, i) => {
    const prev = previous[i];
    return {
      x: prev.x * prevW + point.x * w,
      y: prev.y * prevW + point.y * w,
      z: prev.z * prevW + point.z * w,
    };
  });
}

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  indices: number[],
  width: number,
  height: number,
  fill: string,
  blur = 0
) {
  if (!tracePolygonPath(ctx, points, indices, width, height)) return;
  ctx.save();
  ctx.filter = blur > 0 ? `blur(${blur}px)` : "none";
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.restore();
}

function drawLiner(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  indices: number[],
  width: number,
  height: number,
  color: string
) {
  const pathPoints = toCanvasPathPoints(points, indices, width, height);
  if (pathPoints.length < 2) return;

  const drawPath = () => {
    traceSmoothPath(ctx, pathPoints, false);
  };

  const baseWidth = Math.max(2, width * 0.0032);
  
  // Layer 1: Soft shadow foundation (natural lash line depth)
  ctx.save();
  ctx.filter = "blur(4.5px)";
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.20;
  ctx.lineWidth = baseWidth * 1.8;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  drawPath();
  ctx.stroke();
  ctx.restore();
  
  // Layer 2: Soft diffused base (blended effect)
  ctx.save();
  ctx.filter = "blur(2.8px)";
  ctx.globalCompositeOperation = "soft-light";
  ctx.globalAlpha = 0.32;
  ctx.lineWidth = baseWidth * 1.25;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  drawPath();
  ctx.stroke();
  ctx.restore();

  // Layer 3: Mid-tone pigment (buildable color)
  ctx.save();
  ctx.filter = "blur(1.6px)";
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.42;
  ctx.lineWidth = baseWidth * 0.85;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  drawPath();
  ctx.stroke();
  ctx.restore();
  
  // Layer 4: Defined line (precision application)
  ctx.save();
  ctx.filter = "blur(0.9px)";
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.36;
  ctx.lineWidth = baseWidth * 0.55;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  drawPath();
  ctx.stroke();
  ctx.restore();
  
  // Layer 5: Fine detail accent (like gel liner)
  ctx.save();
  ctx.filter = "blur(0.5px)";
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.26;
  ctx.lineWidth = baseWidth * 0.38;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  drawPath();
  ctx.stroke();
  ctx.restore();
}

function drawProfessionalLips(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  color: string,
  alpha: number,
  lighting = 0.5
) {
  // Adjust opacity based on lighting for realism
  const lightingAdjustment = 0.85 + lighting * 0.3;
  alpha = alpha * lightingAdjustment;
  const lipPath = toCanvasPathPoints(points, LIPS_OUTER, width, height);
  const innerLipPath = toCanvasPathPoints(points, LIPS_INNER, width, height);
  if (lipPath.length < 3) return;

  // Get lip dimensions for gradient positioning
  const lipTop = points[0];  // Cupid's bow center
  const lipBottom = points[17];  // Bottom lip center
  const lipLeft = points[61];  // Left corner
  const lipRight = points[291];  // Right corner
  
  if (!lipTop || !lipBottom || !lipLeft || !lipRight) return;

  const centerX = ((lipLeft.x + lipRight.x) * 0.5) * width;
  const centerY = ((lipTop.y + lipBottom.y) * 0.5) * height;
  const lipWidth = Math.abs(lipRight.x - lipLeft.x) * width;
  const lipHeight = Math.abs(lipBottom.y - lipTop.y) * height;

  // LAYER 1: Soft outer feathering (natural lip liner blend)
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  ctx.filter = "blur(8px)";
  ctx.fillStyle = hexToRgba(color, alpha * 0.18);
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.fill();
  }
  ctx.restore();

  // LAYER 2: Base color application (cream lipstick texture)
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  ctx.filter = "blur(5px)";
  ctx.fillStyle = hexToRgba(color, alpha * 0.35);
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.fill();
  }
  ctx.restore();

  // LAYER 3: Rich color build-up (opaque coverage)
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  ctx.filter = "blur(3px)";
  ctx.fillStyle = hexToRgba(color, alpha * 0.42);
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.fill();
  }
  ctx.restore();

  // LAYER 4: Dimensional shading (corners darker for depth)
  ctx.save();
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.clip();
    
    // Left corner depth
    const leftGradient = ctx.createRadialGradient(
      lipLeft.x * width, lipLeft.y * height, lipWidth * 0.05,
      lipLeft.x * width, lipLeft.y * height, lipWidth * 0.18
    );
    leftGradient.addColorStop(0, hexToRgba(color, alpha * 0.32));
    leftGradient.addColorStop(0.6, hexToRgba(color, alpha * 0.12));
    leftGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(6px)";
    ctx.fillStyle = leftGradient;
    ctx.fillRect(lipLeft.x * width - lipWidth * 0.2, lipLeft.y * height - lipHeight * 0.5, lipWidth * 0.4, lipHeight * 2);
    
    // Right corner depth
    const rightGradient = ctx.createRadialGradient(
      lipRight.x * width, lipRight.y * height, lipWidth * 0.05,
      lipRight.x * width, lipRight.y * height, lipWidth * 0.18
    );
    rightGradient.addColorStop(0, hexToRgba(color, alpha * 0.32));
    rightGradient.addColorStop(0.6, hexToRgba(color, alpha * 0.12));
    rightGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = rightGradient;
    ctx.fillRect(lipRight.x * width - lipWidth * 0.2, lipRight.y * height - lipHeight * 0.5, lipWidth * 0.4, lipHeight * 2);
  }
  ctx.restore();

  // LAYER 5: Soft-light overlay for natural sheen
  ctx.save();
  ctx.globalCompositeOperation = "soft-light";
  ctx.filter = "blur(4px)";
  ctx.fillStyle = hexToRgba(color, alpha * 0.28);
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.fill();
  }
  ctx.restore();

  // LAYER 6: Lip contour definition (subtle edge definition)
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  ctx.filter = "blur(2.5px)";
  ctx.strokeStyle = hexToRgba(color, alpha * 0.25);
  ctx.lineWidth = Math.max(1.5, width * 0.0025);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.stroke();
  }
  ctx.restore();

  // LAYER 7: Center highlight (light catches the center of lips)
  ctx.save();
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.clip();
    
    const centerHighlight = ctx.createRadialGradient(
      centerX, centerY, lipWidth * 0.05,
      centerX, centerY, lipWidth * 0.25
    );
    centerHighlight.addColorStop(0, `rgba(255,245,240,${alpha * 0.22})`);
    centerHighlight.addColorStop(0.5, `rgba(255,240,235,${alpha * 0.10})`);
    centerHighlight.addColorStop(1, "rgba(255,235,230,0)");
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(5px)";
    ctx.fillStyle = centerHighlight;
    ctx.fillRect(centerX - lipWidth * 0.3, centerY - lipHeight * 0.6, lipWidth * 0.6, lipHeight * 1.2);
  }
  ctx.restore();

  // LAYER 8: Cupid's bow highlight (professional gloss)
  const cupidsBowY = lipTop.y * height;
  ctx.save();
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.clip();
    
    const cupidsHighlight = ctx.createRadialGradient(
      centerX, cupidsBowY, 1,
      centerX, cupidsBowY, lipWidth * 0.14
    );
    cupidsHighlight.addColorStop(0, `rgba(255,250,248,${alpha * 0.38})`);
    cupidsHighlight.addColorStop(0.50, `rgba(255,245,240,${alpha * 0.18})`);
    cupidsHighlight.addColorStop(1, "rgba(255,240,235,0)");
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(3px)";
    ctx.fillStyle = cupidsHighlight;
    ctx.fillRect(centerX - lipWidth * 0.18, cupidsBowY - lipHeight * 0.35, lipWidth * 0.36, lipHeight * 0.6);
  }
  ctx.restore();

  // LAYER 9: Bottom lip shine (glossy plump look)
  const bottomLipY = lipBottom.y * height;
  ctx.save();
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.clip();
    
    const bottomShine = ctx.createRadialGradient(
      centerX, bottomLipY - lipHeight * 0.15, lipWidth * 0.03,
      centerX, bottomLipY - lipHeight * 0.15, lipWidth * 0.22
    );
    bottomShine.addColorStop(0, `rgba(255,252,250,${alpha * 0.42})`);
    bottomShine.addColorStop(0.40, `rgba(255,248,245,${alpha * 0.20})`);
    bottomShine.addColorStop(1, "rgba(255,242,238,0)");
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(3.5px)";
    ctx.fillStyle = bottomShine;
    ctx.fillRect(centerX - lipWidth * 0.28, bottomLipY - lipHeight * 0.55, lipWidth * 0.56, lipHeight * 0.7);
  }
  ctx.restore();
  
  // LAYER 9.5: Extra gloss shine (high gloss finish)
  ctx.save();
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.clip();
    
    const extraGloss = ctx.createRadialGradient(
      centerX, centerY, 1,
      centerX, centerY, lipWidth * 0.18
    );
    extraGloss.addColorStop(0, `rgba(255,255,255,${alpha * 0.25})`);
    extraGloss.addColorStop(0.55, `rgba(255,250,248,${alpha * 0.10})`);
    extraGloss.addColorStop(1, "rgba(255,245,242,0)");
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(4px)";
    ctx.fillStyle = extraGloss;
    ctx.fillRect(centerX - lipWidth * 0.22, centerY - lipHeight * 0.4, lipWidth * 0.44, lipHeight * 0.8);
  }
  ctx.restore();

  // LAYER 10: Final saturation boost
  ctx.save();
  ctx.globalCompositeOperation = "soft-light";
  ctx.filter = "blur(3px)";
  ctx.fillStyle = hexToRgba(color, alpha * 0.18);
  if (traceSmoothPath(ctx, lipPath, true)) {
    ctx.fill();
  }
  ctx.restore();

  // FINAL: Protect the mouth interior (cut out the gap between lips)
  if (innerLipPath.length >= 3) {
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.filter = "blur(1px)";
    ctx.fillStyle = "rgba(0,0,0,1)";
    if (traceSmoothPath(ctx, innerLipPath, true)) {
      ctx.fill();
    }
    ctx.restore();
  }
  
  // Add subtle texture for realism (professional quality)
  if (lipPath.length >= 3) {
    ctx.save();
    if (traceSmoothPath(ctx, lipPath, true)) {
      ctx.clip();
      const lipX = Math.floor(Math.min(...lipPath.map(p => p.x)));
      const lipY = Math.floor(Math.min(...lipPath.map(p => p.y)));
      const lipW = Math.ceil(Math.max(...lipPath.map(p => p.x)) - lipX);
      const lipH = Math.ceil(Math.max(...lipPath.map(p => p.y)) - lipY);
      addMakeupTexture(ctx, lipX, lipY, lipW, lipH, alpha * 1.5);
    }
    ctx.restore();
  }
}

function drawProfessionalBlush(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  color: string,
  alpha: number,
  lighting = 0.5
) {
  // Adjust based on lighting - more visible in well-lit areas
  const lightingAdjustment = 0.8 + lighting * 0.4;
  alpha = alpha * lightingAdjustment;
  const left = points[205];
  const right = points[425];
  if (!left || !right) return;
  const radius = Math.max(18, width * 0.052);

  const apply = (center: FacePoint, direction: -1 | 1) => {
    // Professional placement: apple of cheek with natural lift
    const cx = center.x * width + direction * width * 0.018;
    const cy = center.y * height - height * 0.012;
    
    // Layer 1: Ultra-soft outer diffusion (professional airbrush)
    const outerDiffusion = ctx.createRadialGradient(cx, cy, radius * 0.1, cx, cy, radius * 1.25);
    outerDiffusion.addColorStop(0, hexToRgba(color, alpha * 0.18));
    outerDiffusion.addColorStop(0.35, hexToRgba(color, alpha * 0.11));
    outerDiffusion.addColorStop(0.7, hexToRgba(color, alpha * 0.05));
    outerDiffusion.addColorStop(1, "rgba(0,0,0,0)");
    ctx.save();
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(12px)";
    ctx.fillStyle = outerDiffusion;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 1.25, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    // Layer 2: Soft base layer (vibrant buildable color)
    const baseGradient = ctx.createRadialGradient(cx, cy, radius * 0.08, cx, cy, radius * 0.95);
    baseGradient.addColorStop(0, hexToRgba(color, alpha * 0.24));
    baseGradient.addColorStop(0.4, hexToRgba(color, alpha * 0.15));
    baseGradient.addColorStop(0.75, hexToRgba(color, alpha * 0.07));
    baseGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.save();
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(10px)";
    ctx.fillStyle = baseGradient;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.95, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    // Layer 3: Concentrated center (natural flush)
    const centerGradient = ctx.createRadialGradient(cx, cy, radius * 0.06, cx, cy, radius * 0.68);
    centerGradient.addColorStop(0, hexToRgba(color, alpha * 0.28));
    centerGradient.addColorStop(0.45, hexToRgba(color, alpha * 0.17));
    centerGradient.addColorStop(0.85, hexToRgba(color, alpha * 0.07));
    centerGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.save();
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(7px)";
    ctx.fillStyle = centerGradient;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.68, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    // Layer 4: Cheekbone highlight (natural glow)
    const highlightGradient = ctx.createRadialGradient(
      cx - direction * width * 0.010, 
      cy - height * 0.020, 
      1, 
      cx - direction * width * 0.010, 
      cy - height * 0.020, 
      radius * 0.40
    );
    highlightGradient.addColorStop(0, `rgba(255,252,248,${alpha * 0.12})`);
    highlightGradient.addColorStop(0.5, `rgba(255,248,242,${alpha * 0.05})`);
    highlightGradient.addColorStop(1, "rgba(255,245,240,0)");
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(5px)";
    ctx.fillStyle = highlightGradient;
    ctx.beginPath();
    ctx.arc(cx - direction * width * 0.010, cy - height * 0.020, radius * 0.40, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Layer 5: Additional soft overlay for natural blend
    ctx.save();
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(8px)";
    ctx.fillStyle = hexToRgba(color, alpha * 0.14);
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.82, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  apply(left, -1);
  apply(right, 1);
}

function drawProfessionalEyeshadow(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  color: string,
  alpha: number,
  lighting = 0.5
) {
  // Eyeshadow appears more vibrant in good lighting
  const lightingAdjustment = 0.85 + lighting * 0.3;
  alpha = alpha * lightingAdjustment;
  const distance = (a: CanvasPoint, b: CanvasPoint) => Math.hypot(a.x - b.x, a.y - b.y);

  const drawOne = (upperLid: number[], brow: number[], eyeSocket: number[]) => {
    // Get key eye landmarks
    const outerCorner = points[upperLid[0]];
    const innerCorner = points[upperLid[upperLid.length - 1]];
    const lidMid = points[upperLid[Math.floor(upperLid.length / 2)]];
    
    if (!outerCorner || !innerCorner || !lidMid) return;

    const eyeSocketPath = toCanvasPathPoints(points, eyeSocket, width, height);
    const lidPath = toCanvasPathPoints(points, upperLid, width, height);
    const browPath = toCanvasPathPoints(points, brow, width, height);

    // Calculate eye geometry
    const outerX = outerCorner.x * width;
    const outerY = outerCorner.y * height;
    const innerX = innerCorner.x * width;
    const innerY = innerCorner.y * height;
    const midX = lidMid.x * width;
    const midY = lidMid.y * height;
    
    const eyeWidth = Math.abs(outerX - innerX);
    const eyeHeight = Math.abs(midY - ((outerY + innerY) / 2));
    const centerX = (outerX + innerX) / 2;
    const centerY = midY;
    
    // Calculate eye angle for natural application
    const eyeAngle = Math.atan2(innerY - outerY, innerX - outerX);
    
    // Professional zones - keep eyeshadow well below brow
    const creaseY = midY - eyeHeight * 1.0;
    const outerV_X = outerX + (outerX > centerX ? eyeWidth * 0.05 : -eyeWidth * 0.05);
    const outerV_Y = creaseY;

    ctx.save();
    
    // LAYER 1: Soft base wash following eye shape
    ctx.translate(centerX, centerY);
    ctx.rotate(eyeAngle);
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(14px)";
    const washGradient = ctx.createRadialGradient(0, 0, eyeWidth * 0.1, 0, -eyeHeight * 0.4, eyeWidth * 0.50);
    washGradient.addColorStop(0, hexToRgba(color, alpha * 0.18));
    washGradient.addColorStop(0.6, hexToRgba(color, alpha * 0.10));
    washGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = washGradient;
    ctx.beginPath();
    ctx.ellipse(0, -eyeHeight * 0.2, eyeWidth * 0.52, eyeHeight * 1.4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // LAYER 2: Mobile lid color (main eyeshadow on the lid)
    ctx.translate(centerX, centerY);
    ctx.rotate(eyeAngle);
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(7px)";
    const lidGradient = ctx.createRadialGradient(0, 0, eyeWidth * 0.08, 0, -eyeHeight * 0.3, eyeWidth * 0.46);
    lidGradient.addColorStop(0, hexToRgba(color, alpha * 0.50));
    lidGradient.addColorStop(0.45, hexToRgba(color, alpha * 0.35));
    lidGradient.addColorStop(0.80, hexToRgba(color, alpha * 0.15));
    lidGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = lidGradient;
    ctx.beginPath();
    ctx.ellipse(0, -eyeHeight * 0.15, eyeWidth * 0.48, eyeHeight * 1.2, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // LAYER 3: Crease definition (contoured depth)
    ctx.translate(centerX, creaseY);
    ctx.rotate(eyeAngle);
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(8px)";
    const creaseGradient = ctx.createRadialGradient(0, 0, eyeWidth * 0.08, 0, 0, eyeWidth * 0.40);
    creaseGradient.addColorStop(0, hexToRgba(color, alpha * 0.55));
    creaseGradient.addColorStop(0.40, hexToRgba(color, alpha * 0.38));
    creaseGradient.addColorStop(0.75, hexToRgba(color, alpha * 0.18));
    creaseGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = creaseGradient;
    ctx.beginPath();
    ctx.ellipse(0, 0, eyeWidth * 0.42, eyeHeight * 0.9, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // LAYER 4: Outer V depth (professional smoky technique)
    ctx.translate(outerV_X, outerV_Y);
    ctx.rotate(eyeAngle);
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(9px)";
    const outerVGradient = ctx.createRadialGradient(0, 0, eyeWidth * 0.06, 0, 0, eyeWidth * 0.32);
    outerVGradient.addColorStop(0, hexToRgba(color, alpha * 0.60));
    outerVGradient.addColorStop(0.35, hexToRgba(color, alpha * 0.42));
    outerVGradient.addColorStop(0.70, hexToRgba(color, alpha * 0.20));
    outerVGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = outerVGradient;
    ctx.beginPath();
    ctx.ellipse(0, 0, eyeWidth * 0.35, eyeHeight * 0.85, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // LAYER 5: Soft-light overlay for luminosity
    ctx.translate(centerX, centerY);
    ctx.rotate(eyeAngle);
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(11px)";
    const overlayGradient = ctx.createRadialGradient(0, -eyeHeight * 0.1, eyeWidth * 0.1, 0, -eyeHeight * 0.4, eyeWidth * 0.48);
    overlayGradient.addColorStop(0, hexToRgba(color, alpha * 0.25));
    overlayGradient.addColorStop(0.50, hexToRgba(color, alpha * 0.15));
    overlayGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = overlayGradient;
    ctx.beginPath();
    ctx.ellipse(0, -eyeHeight * 0.2, eyeWidth * 0.50, eyeHeight * 1.3, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // LAYER 6: Color intensity on lid center
    ctx.translate(centerX, centerY);
    ctx.rotate(eyeAngle);
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(6px)";
    const intensityGradient = ctx.createRadialGradient(0, 0, eyeWidth * 0.08, 0, -eyeHeight * 0.1, eyeWidth * 0.33);
    intensityGradient.addColorStop(0, hexToRgba(color, alpha * 0.38));
    intensityGradient.addColorStop(0.55, hexToRgba(color, alpha * 0.22));
    intensityGradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = intensityGradient;
    ctx.beginPath();
    ctx.ellipse(0, -eyeHeight * 0.05, eyeWidth * 0.35, eyeHeight * 0.8, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // LAYER 7: Lash line definition
    if (lidPath.length > 2) {
      ctx.globalCompositeOperation = "multiply";
      ctx.filter = "blur(5px)";
      ctx.strokeStyle = hexToRgba(color, alpha * 0.28);
      ctx.lineWidth = eyeWidth * 0.08;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      if (traceSmoothPath(ctx, lidPath, false)) {
        ctx.stroke();
      }
    }

    // LAYER 8: Inner corner highlight (professional brightening)
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(4.5px)";
    const highlightGradient = ctx.createRadialGradient(innerX, innerY, 1, innerX, innerY, eyeWidth * 0.20);
    highlightGradient.addColorStop(0, `rgba(255,248,242,${alpha * 0.22})`);
    highlightGradient.addColorStop(0.50, `rgba(255,242,235,${alpha * 0.12})`);
    highlightGradient.addColorStop(1, "rgba(255,238,230,0)");
    ctx.fillStyle = highlightGradient;
    ctx.beginPath();
    ctx.arc(innerX, innerY, eyeWidth * 0.20, 0, Math.PI * 2);
    ctx.fill();

    // LAYER 9: Center lid shimmer (glamorous effect)
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(3.5px)";
    const shimmerGradient = ctx.createRadialGradient(centerX, centerY, 1, centerX, centerY, eyeWidth * 0.20);
    shimmerGradient.addColorStop(0, `rgba(255,250,245,${alpha * 0.20})`);
    shimmerGradient.addColorStop(0.45, `rgba(255,245,238,${alpha * 0.10})`);
    shimmerGradient.addColorStop(1, "rgba(255,240,232,0)");
    ctx.fillStyle = shimmerGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, eyeWidth * 0.20, 0, Math.PI * 2);
    ctx.fill();
    
    // LAYER 10: Subtle sparkle (subtle metallic finish)
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(2.5px)";
    const sparkleGradient = ctx.createRadialGradient(centerX, centerY, 1, centerX, centerY - eyeHeight * 0.1, eyeWidth * 0.15);
    sparkleGradient.addColorStop(0, `rgba(255,255,255,${alpha * 0.18})`);
    sparkleGradient.addColorStop(0.60, `rgba(255,248,242,${alpha * 0.08})`);
    sparkleGradient.addColorStop(1, "rgba(255,242,235,0)");
    ctx.fillStyle = sparkleGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY - eyeHeight * 0.05, eyeWidth * 0.15, 0, Math.PI * 2);
    ctx.fill();

    // PROTECT: Eyeball area (never put makeup on the eye)
    if (eyeSocketPath.length >= 3) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.filter = "blur(2px)";
      ctx.fillStyle = "rgba(0,0,0,1)";
      if (traceSmoothPath(ctx, eyeSocketPath, true)) {
        ctx.fill();
      }
    }

    // PROTECT: Eyebrow area - soft natural protection
    if (browPath.length >= 3) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      
      // Soft graduated protection - multiple passes with increasing blur
      for (let pass = 0; pass < 3; pass++) {
        const blurAmount = 5 + pass * 2;
        const lineWidth = eyeWidth * (0.05 + pass * 0.015);
        const opacity = 0.3 + pass * 0.15;
        
        ctx.filter = `blur(${blurAmount}px)`;
        ctx.strokeStyle = `rgba(0,0,0,${opacity})`;
        ctx.lineWidth = lineWidth;
        
        if (traceSmoothPath(ctx, browPath, false)) {
          ctx.stroke();
        }
      }
      
      // Gentle fade at the ends to prevent harsh edges
      if (browPath.length > 0) {
        ctx.filter = "blur(8px)";
        ctx.fillStyle = "rgba(0,0,0,0.2)";
        
        // Start of brow
        ctx.beginPath();
        ctx.arc(browPath[0].x, browPath[0].y, eyeWidth * 0.06, 0, Math.PI * 2);
        ctx.fill();
        
        // End of brow  
        ctx.beginPath();
        ctx.arc(browPath[browPath.length - 1].x, browPath[browPath.length - 1].y, eyeWidth * 0.06, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
  };

  drawOne(LEFT_LINER, LEFT_BROW, LEFT_EYE_SOCKET);
  drawOne(RIGHT_LINER, RIGHT_BROW, RIGHT_EYE_SOCKET);
}

function drawUnderEyeConcealer(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  alpha: number
) {
  const drawOne = (outerIndex: number, innerIndex: number, lowerIndex: number) => {
    const outer = points[outerIndex];
    const inner = points[innerIndex];
    const lower = points[lowerIndex];
    if (!outer || !inner || !lower) return;

    const outerX = outer.x * width;
    const outerY = outer.y * height;
    const innerX = inner.x * width;
    const innerY = inner.y * height;
    const lowerX = lower.x * width;
    const lowerY = lower.y * height;

    const cx = (outerX + innerX) * 0.5 * 0.88 + lowerX * 0.12;
    const cy = (outerY + innerY) * 0.5 * 0.35 + lowerY * 0.65;
    const eyeWidth = Math.hypot(innerX - outerX, innerY - outerY);
    const angle = Math.atan2(innerY - outerY, innerX - outerX);
    const rx = Math.max(10, eyeWidth * 0.52);
    const ry = Math.max(7, eyeWidth * 0.28);

    ctx.save();
    ctx.translate(cx, cy + ry * 0.58);
    ctx.rotate(angle);
    
    // Layer 1: Ultra-soft outer brightening (natural radiance)
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(12px)";
    ctx.fillStyle = `rgba(245, 220, 200, ${alpha * 0.14})`;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx * 1.10, ry * 1.15, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Layer 2: Broad brightening base (seamless blend)
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(9px)";
    ctx.fillStyle = `rgba(242, 215, 195, ${alpha * 0.18})`;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx * 0.95, ry * 1.05, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Layer 3: Color correction (peachy warmth)
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(7px)";
    ctx.fillStyle = `rgba(238, 200, 175, ${alpha * 0.20})`;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx * 0.85, ry * 0.90, 0, 0, Math.PI * 2);
    ctx.fill();

    // Layer 4: Concentrated coverage on darkest area
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(5px)";
    ctx.fillStyle = `rgba(232, 188, 165, ${alpha * 0.16})`;
    ctx.beginPath();
    ctx.ellipse(0, ry * 0.18, rx * 0.60, ry * 0.48, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Layer 5: Inner corner highlight (brightening technique)
    ctx.globalCompositeOperation = "screen";
    ctx.filter = "blur(4px)";
    ctx.fillStyle = `rgba(252, 245, 238, ${alpha * 0.10})`;
    ctx.beginPath();
    ctx.ellipse(rx * 0.35, -ry * 0.12, rx * 0.25, ry * 0.28, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Layer 6: Additional soft-light for natural finish
    ctx.globalCompositeOperation = "soft-light";
    ctx.filter = "blur(8px)";
    ctx.fillStyle = `rgba(240, 210, 190, ${alpha * 0.12})`;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx * 0.80, ry * 0.85, 0, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.restore();
  };

  drawOne(LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_LOWER);
  drawOne(RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_LOWER);
}

function protectEyeInterior(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  outerIndex: number,
  innerIndex: number,
  upperIndex: number,
  lowerIndex: number,
  scaleX = 0.9,
  scaleY = 0.58
) {
  const outer = points[outerIndex];
  const inner = points[innerIndex];
  const upper = points[upperIndex];
  const lower = points[lowerIndex];
  if (!outer || !inner || !upper || !lower) return;

  const outerX = outer.x * width;
  const outerY = outer.y * height;
  const innerX = inner.x * width;
  const innerY = inner.y * height;
  const upperX = upper.x * width;
  const upperY = upper.y * height;
  const lowerX = lower.x * width;
  const lowerY = lower.y * height;

  const centerX = (outerX + innerX + upperX + lowerX) * 0.25;
  const centerY = (outerY + innerY + upperY + lowerY) * 0.25;
  const eyeWidth = Math.hypot(innerX - outerX, innerY - outerY);
  const eyeHeight = Math.hypot(lowerX - upperX, lowerY - upperY);
  const angle = Math.atan2(innerY - outerY, innerX - outerX);
  const rx = Math.max(5, eyeWidth * 0.5 * scaleX);
  const ry = Math.max(3, eyeHeight * 0.5 * scaleY);

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(angle);
  ctx.globalCompositeOperation = "destination-out";
  ctx.filter = "blur(0.8px)";
  ctx.fillStyle = "rgba(0,0,0,1)";
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function protectEyeRegionByPath(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  indices: number[]
) {
  const eyePath = toCanvasPathPoints(points, indices, width, height);
  if (eyePath.length < 3) return;

  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.filter = "blur(0.8px)";
  ctx.fillStyle = "rgba(0,0,0,1)";
  if (traceSmoothPath(ctx, eyePath, true)) {
    ctx.fill();
  }

  // Expand protection around eyelid margins so blurred pigments cannot bleed in.
  ctx.filter = "blur(1.6px)";
  ctx.lineWidth = Math.max(3, width * 0.006);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(0,0,0,1)";
  if (traceSmoothPath(ctx, eyePath, true)) {
    ctx.stroke();
  }
  ctx.restore();
}

function protectBrowRegionByPath(
  ctx: CanvasRenderingContext2D,
  points: FacePoint[],
  width: number,
  height: number,
  browIndices: number[]
) {
  const browPath = toCanvasPathPoints(points, browIndices, width, height);
  if (browPath.length < 3) return;

  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.filter = "blur(1.2px)";
  ctx.fillStyle = "rgba(0,0,0,1)";
  if (traceSmoothPath(ctx, browPath, true)) {
    ctx.fill();
  }

  // Slightly widen the protection to keep shadow fade below the brow.
  ctx.filter = "blur(1.8px)";
  ctx.lineWidth = Math.max(3, width * 0.007);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0,0,0,1)";
  if (traceSmoothPath(ctx, browPath, true)) {
    ctx.stroke();
  }
  ctx.restore();
}

function getCameraErrorMessage(error: unknown) {
  const fallback = "Camera unavailable. Check permission and retry.";
  if (!(error instanceof DOMException)) return fallback;
  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission denied. Enable camera in browser/webview settings.";
    case "NotFoundError":
      return "No camera found on this device.";
    case "NotReadableError":
      return "Camera is in use by another app.";
    default:
      return fallback;
  }
}


export default function MakeupStudio() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const beautyCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const [status, setStatus] = useState("Initializing camera...");
  const [isRunning, setIsRunning] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [smoothAmount, setSmoothAmount] = useState(0.58);
  const [toneAmount, setToneAmount] = useState(0.42);
  const [sharpenAmount, setSharpenAmount] = useState(0.15);
  const [activeStyleId, setActiveStyleId] = useState(STYLES[0].id);
  const [masterIntensity, setMasterIntensity] = useState(0.95);

  const activeStyle = useMemo(() => STYLES.find((style) => style.id === activeStyleId) ?? STYLES[0], [activeStyleId]);

  const smoothAmountRef = useRef(smoothAmount);
  const toneAmountRef = useRef(toneAmount);
  const sharpenAmountRef = useRef(sharpenAmount);
  const activeStyleRef = useRef(activeStyle);
  const masterIntensityRef = useRef(masterIntensity);

  useEffect(() => {
    smoothAmountRef.current = smoothAmount;
  }, [smoothAmount]);
  useEffect(() => {
    toneAmountRef.current = toneAmount;
  }, [toneAmount]);
  useEffect(() => {
    sharpenAmountRef.current = sharpenAmount;
  }, [sharpenAmount]);
  useEffect(() => {
    activeStyleRef.current = activeStyle;
  }, [activeStyle]);
  useEffect(() => {
    masterIntensityRef.current = masterIntensity;
  }, [masterIntensity]);

  useEffect(() => {
    let disposed = false;
    let rafId: number | null = null;
    let stream: MediaStream | null = null;
    let landmarker: FaceLandmarker | null = null;
    let beautyRenderer: BeautyRenderer | null = null;

    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = 1024; // Higher resolution for professional quality
    maskCanvas.height = 1024;
    const maskCtx = maskCanvas.getContext("2d", { 
      willReadFrequently: false,
      alpha: true,
      desynchronized: false
    });
    
    if (maskCtx) {
      maskCtx.imageSmoothingEnabled = true;
      maskCtx.imageSmoothingQuality = "high";
    }

    const run = async () => {
      try {
        const video = videoRef.current;
        const beautyCanvas = beautyCanvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        if (!video || !beautyCanvas || !overlayCanvas || !maskCtx) return;

        // Request high-quality camera settings (Snapchat/Instagram quality)
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1920, min: 1280 },
            height: { ideal: 1080, min: 720 },
            frameRate: { ideal: 60, min: 30 },
            aspectRatio: { ideal: 16 / 9 },
          },
          audio: false,
        });

        if (disposed) return;

        video.srcObject = stream;
        await video.play();

        beautyRenderer = createBeautyRenderer(beautyCanvas);

        const overlayCtx = overlayCanvas.getContext("2d", {
          alpha: true,
          desynchronized: false,
          willReadFrequently: false
        });
        if (!overlayCtx) throw new Error("Overlay canvas 2D context unavailable.");
        
        // Enable high-quality rendering
        overlayCtx.imageSmoothingEnabled = true;
        overlayCtx.imageSmoothingQuality = "high";

        setStatus("Loading face tracking...");
        const fileset = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });

        setStatus("Live: professional makeup styles");
        setIsRunning(true);

        let lastInferenceMs = 0;
        let latestPoints: FacePoint[] | null = null;
        let smoothedPoints: FacePoint[] | null = null;
        let lastVideoTime = -1;
        let detectErrorCooldownUntil = 0;
        
        // Professional-grade tracking components
        const kalmanFilter = new LandmarkKalmanFilter(478); // MediaPipe has 478 landmarks
        const motionDetector = new MotionDetector();
        const dirtyRegionTracker = new DirtyRegionTracker();
        const smoothSmoother = new ValueSmoother(smoothAmountRef.current, 0.12);
        const toneSmoother = new ValueSmoother(toneAmountRef.current, 0.12);
        const sharpenSmoother = new ValueSmoother(sharpenAmountRef.current, 0.12);

        const resizeToVideo = () => {
          const width = video.videoWidth;
          const height = video.videoHeight;
          if (!width || !height) return;
          
          // Set aspect ratio explicitly for iOS compatibility
          const aspectRatio = width / height;
          beautyCanvas.style.aspectRatio = `${aspectRatio}`;
          
          beautyRenderer?.resize(width, height);
          if (overlayCanvas.width !== width || overlayCanvas.height !== height) {
            overlayCanvas.width = width;
            overlayCanvas.height = height;
            // Restore quality settings after resize
            overlayCtx.imageSmoothingEnabled = true;
            overlayCtx.imageSmoothingQuality = "high";
            overlayCtx.globalCompositeOperation = "source-over";
          }
        };

        const updateMask = (points: FacePoint[] | null) => {
          maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          if (!points) return;
          const start = points[FACE_OVAL[0]];
          if (!start) return;

          maskCtx.save();
          maskCtx.beginPath();
          maskCtx.moveTo(start.x * maskCanvas.width, start.y * maskCanvas.height);
          for (let i = 1; i < FACE_OVAL.length; i += 1) {
            const p = points[FACE_OVAL[i]];
            if (!p) continue;
            maskCtx.lineTo(p.x * maskCanvas.width, p.y * maskCanvas.height);
          }
          maskCtx.closePath();
          maskCtx.fillStyle = "rgba(255,255,255,0.98)";
          maskCtx.fill();
          maskCtx.restore();

          // Multi-pass blur for ultra-smooth feathering (Instagram quality)
          maskCtx.save();
          maskCtx.filter = "blur(8px)";
          maskCtx.drawImage(maskCanvas, 0, 0);
          maskCtx.restore();
          
          maskCtx.save();
          maskCtx.filter = "blur(5px)";
          maskCtx.drawImage(maskCanvas, 0, 0);
          maskCtx.restore();
          
          maskCtx.save();
          maskCtx.filter = "blur(3px)";
          maskCtx.drawImage(maskCanvas, 0, 0);
          maskCtx.restore();
        };

        const renderMakeup = (points: FacePoint[] | null) => {
          const width = overlayCanvas.width;
          const height = overlayCanvas.height;
          
          // Clear canvas completely
          overlayCtx.clearRect(0, 0, width, height);
          
          if (!points || !width || !height) {
            return;
          }

          // Reset to default state
          overlayCtx.globalCompositeOperation = "source-over";
          overlayCtx.globalAlpha = 1.0;
          overlayCtx.filter = "none";

          // Match landmark coordinates to current video presentation:
          // mirrored horizontally for selfie view.
          const displayPoints = points.map((point) => ({
            ...point,
            x: 1 - point.x,
            y: point.y,
          }));

          // Detect lighting for adaptive makeup application (Instagram/Snapchat quality)
          const lighting = detectFaceLighting(overlayCtx, displayPoints, width, height);

          const style = activeStyleRef.current;
          const intensity = masterIntensityRef.current;
          // Professional-quality filter: vibrant, polished, professional
          const lipAlpha = Math.min(0.95, Math.max(0.45, style.lipstickOpacity * intensity * 1.10));
          const eyeAlpha = Math.min(0.98, Math.max(0.40, style.eyeshadowOpacity * intensity * 1.15));
          const blushAlpha = Math.min(0.82, Math.max(0.25, style.blushOpacity * intensity * 0.75));
          const linerAlpha = Math.min(0.95, Math.max(0.40, style.linerOpacity * intensity * 0.95));
          const concealerAlpha = Math.min(0.70, 0.28 + intensity * 0.30);

          drawUnderEyeConcealer(overlayCtx, displayPoints, width, height, concealerAlpha);
          drawProfessionalEyeshadow(overlayCtx, displayPoints, width, height, style.eyeshadow, eyeAlpha, lighting);
          drawProfessionalLips(overlayCtx, displayPoints, width, height, style.lipstick, lipAlpha, lighting);
          drawPolygon(
            overlayCtx,
            displayPoints,
            LEFT_CHEEK,
            width,
            height,
            hexToRgba(style.blush, blushAlpha * 0.24),
            14
          );
          drawPolygon(
            overlayCtx,
            displayPoints,
            RIGHT_CHEEK,
            width,
            height,
            hexToRgba(style.blush, blushAlpha * 0.24),
            14
          );
          drawProfessionalBlush(overlayCtx, displayPoints, width, height, style.blush, blushAlpha, lighting);
          drawLiner(overlayCtx, displayPoints, LEFT_LINER, width, height, hexToRgba(style.liner, linerAlpha));
          drawLiner(overlayCtx, displayPoints, RIGHT_LINER, width, height, hexToRgba(style.liner, linerAlpha));
          protectEyeInterior(
            overlayCtx,
            displayPoints,
            width,
            height,
            LEFT_EYE_OUTER,
            LEFT_EYE_INNER,
            159,
            LEFT_EYE_LOWER,
            0.94,
            0.62
          );
          protectEyeInterior(
            overlayCtx,
            displayPoints,
            width,
            height,
            RIGHT_EYE_OUTER,
            RIGHT_EYE_INNER,
            386,
            RIGHT_EYE_LOWER,
            0.9,
            0.58
          );
          protectEyeRegionByPath(overlayCtx, displayPoints, width, height, LEFT_EYE_SOCKET);
          protectEyeRegionByPath(overlayCtx, displayPoints, width, height, RIGHT_EYE_SOCKET);
          protectBrowRegionByPath(overlayCtx, displayPoints, width, height, LEFT_BROW);
          protectBrowRegionByPath(overlayCtx, displayPoints, width, height, RIGHT_BROW);
        };

        const animate = (nowMs: number) => {
          if (disposed) return;

          if (video.readyState >= 2) {
            resizeToVideo();

            // 60fps inference for Instagram/Snapchat quality (16ms instead of 33ms)
            if (video.currentTime !== lastVideoTime && nowMs - lastInferenceMs >= 16 && landmarker) {
              lastVideoTime = video.currentTime;
              try {
                const result = landmarker.detectForVideo(video, nowMs);
                latestPoints = (result.faceLandmarks?.[0] as FacePoint[] | undefined) ?? null;
                
                if (latestPoints) {
                  // Apply Kalman filtering for ultra-smooth tracking
                  const kalmanFiltered = kalmanFilter.filter(latestPoints);
                  
                  // Detect motion for adaptive smoothing
                  const motion = motionDetector.detectMotion(kalmanFiltered);
                  
                  // Adaptive smoothing based on motion
                  smoothedPoints = smoothLandmarksAdaptive(smoothedPoints, kalmanFiltered, motion);
                } else {
                  smoothedPoints = null;
                }
                
                setFaceDetected(latestPoints !== null);
                lastInferenceMs = nowMs;
              } catch (detectError) {
                latestPoints = null;
                smoothedPoints = null;
                if (nowMs >= detectErrorCooldownUntil) {
                  detectErrorCooldownUntil = nowMs + 1200;
                  const message = detectError instanceof Error ? detectError.message : "Landmarker failed";
                  setStatus(`Tracking recovering: ${message}`);
                }
              }
            }

            // Update temporal smoothers for settings
            smoothSmoother.setTarget(smoothAmountRef.current);
            toneSmoother.setTarget(toneAmountRef.current);
            sharpenSmoother.setTarget(sharpenAmountRef.current);

            // Always render beauty filter (it's GPU-based and fast)
            beautyRenderer?.render(
              video,
              maskCanvas,
              smoothSmoother.update(),
              toneSmoother.update(),
              sharpenSmoother.update()
            );

            // Performance optimization: only render makeup when needed
            if (
              dirtyRegionTracker.shouldRender(
                smoothedPoints,
                activeStyleRef.current.id,
                masterIntensityRef.current
              )
            ) {
              updateMask(smoothedPoints);
              renderMakeup(smoothedPoints);
            }
          }

          rafId = window.requestAnimationFrame(animate);
        };

        rafId = window.requestAnimationFrame(animate);
      } catch (error) {
        if (!disposed) {
          setStatus(getCameraErrorMessage(error));
          setIsRunning(false);
        }
      }
    };

    run().catch((error) => {
      if (!disposed) {
        setStatus(error instanceof Error ? error.message : "Failed to initialize.");
        setIsRunning(false);
      }
    });

    return () => {
      disposed = true;
      if (rafId !== null) window.cancelAnimationFrame(rafId);
      landmarker?.close();
      beautyRenderer?.dispose();
      stream?.getTracks().forEach((track) => track.stop());
      setIsRunning(false);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950/30 to-slate-900">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 sm:py-5">
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white sm:text-2xl">
              <span className="bg-gradient-to-r from-fuchsia-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
                ham make-up studio
              </span>
              <span className="ml-2 hidden rounded-lg bg-gradient-to-r from-purple-500/20 to-pink-500/20 px-2 py-0.5 text-[10px] font-semibold text-pink-300 ring-1 ring-pink-400/30 sm:ml-3 sm:inline-block sm:px-3 sm:py-1 sm:text-xs">
                Snapchat/Instagram Quality
              </span>
            </h1>
            <p className="mt-0.5 text-[10px] text-slate-400 sm:mt-1 sm:text-sm">60fps • Kalman Filtering • AI Skin Detection</p>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex items-center gap-1.5 sm:gap-2">
              <div className={`h-2 w-2 rounded-full ${isRunning ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
              <span className="text-xs font-medium text-slate-300 sm:text-sm">
                {isRunning ? "Live" : "Loading"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl gap-4 px-3 py-4 sm:gap-6 sm:px-6 sm:py-6 lg:grid-cols-[1.5fr_1fr] lg:gap-8 lg:py-8">
        {/* Video Section */}
        <section className="space-y-3 sm:space-y-4">
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-2 shadow-2xl backdrop-blur-xl sm:rounded-3xl sm:p-4">
            <div className="relative overflow-hidden rounded-xl border border-white/10 bg-black shadow-inner sm:rounded-2xl">
              <video ref={videoRef} className="pointer-events-none absolute inset-0 h-full w-full opacity-0 object-contain" playsInline muted autoPlay />
              <canvas ref={beautyCanvasRef} className="block h-auto w-full" />
              <canvas ref={overlayCanvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
              
              {/* Overlay UI */}
              <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between p-2 sm:p-5">
                <div className="rounded-lg border border-white/20 bg-black/40 px-2 py-1 backdrop-blur-md sm:rounded-2xl sm:px-4 sm:py-2">
                  <p className="text-[10px] font-medium text-white/90 sm:text-xs">Live Preview</p>
                </div>
                <div className="flex items-center gap-1.5 sm:gap-3">
                  {faceDetected && (
                    <div className="rounded-lg border border-emerald-400/30 bg-emerald-500/20 px-2 py-1 backdrop-blur-md sm:rounded-2xl sm:px-4 sm:py-2">
                      <p className="text-[10px] font-medium text-emerald-300 sm:text-xs">Face Detected</p>
                    </div>
                  )}
                  <div className="hidden rounded-2xl border border-white/20 bg-black/40 px-4 py-2 backdrop-blur-md sm:block">
                    <p className="text-xs font-medium text-white/90">{activeStyle.name}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Status Bar */}
          <div className="flex items-center justify-between rounded-xl border border-white/10 bg-gradient-to-r from-slate-900/80 to-slate-800/80 px-3 py-2 backdrop-blur-sm sm:rounded-2xl sm:px-5 sm:py-3">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-fuchsia-500 to-pink-500 sm:h-8 sm:w-8">
                <svg className="h-3.5 w-3.5 text-white sm:h-4 sm:w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium text-white sm:text-sm">{status}</p>
                <p className="text-[10px] text-slate-400 sm:text-xs">Instagram Quality • 60fps Tracking</p>
              </div>
            </div>
          </div>
        </section>

        {/* Controls Panel */}
        <aside className="space-y-4 sm:space-y-6">
          {/* Makeup Styles */}
          <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-4 shadow-2xl backdrop-blur-xl sm:rounded-3xl sm:p-6">
            <div className="mb-4 flex items-center justify-between sm:mb-5">
              <h2 className="text-base font-semibold text-white sm:text-lg">Makeup Styles</h2>
              <div className="rounded-lg bg-fuchsia-500/20 px-2 py-0.5 text-[10px] font-medium text-fuchsia-300 sm:px-3 sm:py-1 sm:text-xs">
                {STYLES.length} Looks
              </div>
            </div>

            <div className="space-y-2 sm:space-y-2.5">
              {STYLES.map((style) => (
                <button
                  key={style.id}
                  type="button"
                  onClick={() => setActiveStyleId(style.id)}
                  className={`group relative w-full overflow-hidden rounded-lg border p-3 text-left transition-all duration-300 active:scale-[0.98] sm:rounded-xl sm:p-4 ${
                    activeStyleId === style.id
                      ? "border-fuchsia-400/60 bg-gradient-to-r from-fuchsia-500/20 via-pink-500/15 to-rose-500/10 shadow-lg shadow-fuchsia-500/20"
                      : "border-white/10 bg-white/5 hover:border-white/20 hover:bg-white/10"
                  }`}
                >
                  {activeStyleId === style.id && (
                    <div className="absolute inset-0 bg-gradient-to-r from-fuchsia-500/10 to-transparent opacity-50" />
                  )}
                  
                  <div className="relative flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-1.5 sm:gap-2">
                        <p className="text-sm font-semibold text-white sm:text-base">{style.name}</p>
                        {activeStyleId === style.id && (
                          <svg className="h-3.5 w-3.5 text-fuchsia-400 sm:h-4 sm:w-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                        )}
                      </div>
                      <div className="mt-2 flex gap-1.5 sm:mt-2.5 sm:gap-2">
                        {[
                          { color: style.lipstick, label: "Lip" },
                          { color: style.eyeshadow, label: "Eye" },
                          { color: style.blush, label: "Blush" },
                          { color: style.liner, label: "Liner" }
                        ].map((swatch) => (
                          <div key={`${style.id}-${swatch.color}`} className="flex flex-col items-center gap-0.5 sm:gap-1">
                            <span
                              className="h-4 w-4 rounded-full border-2 border-white/40 shadow-lg ring-2 ring-black/20 sm:h-5 sm:w-5"
                              style={{ backgroundColor: swatch.color }}
                            />
                            <span className="text-[8px] font-medium text-slate-400 sm:text-[9px]">{swatch.label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Intensity Controls */}
          <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-4 shadow-2xl backdrop-blur-xl sm:rounded-3xl sm:p-6">
            <div className="mb-4 flex items-center justify-between sm:mb-5">
              <h2 className="text-base font-semibold text-white sm:text-lg">Style Intensity</h2>
              <div className="rounded-lg bg-gradient-to-r from-purple-500/10 to-pink-500/10 px-2 py-0.5 text-[10px] font-medium text-pink-300 ring-1 ring-pink-400/20 sm:px-2 sm:py-1 sm:text-xs">
                ✨ Enhanced
              </div>
            </div>
            
            <div className="space-y-4 sm:space-y-6">
              <div>
                <div className="mb-2 flex items-center justify-between sm:mb-3">
                  <label className="flex items-center gap-1.5 text-xs font-medium text-slate-300 sm:gap-2 sm:text-sm">
                    <svg className="h-3.5 w-3.5 text-fuchsia-400 sm:h-4 sm:w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                    </svg>
                    Makeup Intensity
                  </label>
                  <span className="rounded-lg bg-fuchsia-500/20 px-2 py-0.5 text-[10px] font-bold text-fuchsia-300 sm:px-2.5 sm:py-1 sm:text-xs">
                    {Math.round(masterIntensity * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0.2}
                  max={1}
                  step={0.02}
                  value={masterIntensity}
                  onChange={(event) => setMasterIntensity(Number(event.target.value))}
                  className="h-2.5 w-full cursor-pointer appearance-none rounded-full bg-gradient-to-r from-slate-700 to-slate-600 accent-fuchsia-400 sm:h-2 [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:w-6 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-fuchsia-400 [&::-webkit-slider-thumb]:to-pink-400 [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:ring-2 [&::-webkit-slider-thumb]:ring-fuchsia-300/50 sm:[&::-webkit-slider-thumb]:h-5 sm:[&::-webkit-slider-thumb]:w-5"
                />
              </div>
            </div>
          </div>

          {/* Skin Enhancement */}
          <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-4 shadow-2xl backdrop-blur-xl sm:rounded-3xl sm:p-6">
            <div className="mb-4 flex items-center justify-between sm:mb-5">
              <h2 className="text-base font-semibold text-white sm:text-lg">Beauty Filter</h2>
              <div className="flex items-center gap-1.5 sm:gap-2">
                <div className="rounded-lg bg-purple-500/20 px-1.5 py-0.5 text-[10px] font-medium text-purple-300 sm:px-2 sm:py-1 sm:text-xs">
                  AI Enhanced
                </div>
                <div className="rounded-lg bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300 sm:px-2 sm:py-1 sm:text-xs">
                  60fps
                </div>
              </div>
            </div>
            
            <div className="space-y-4 sm:space-y-5">
              <div>
                <div className="mb-2 flex items-center justify-between sm:mb-2.5">
                  <label className="text-xs font-medium text-slate-300 sm:text-sm">✨ Skin Smoothing</label>
                  <span className="rounded-lg bg-purple-500/20 px-2 py-0.5 text-[10px] font-bold text-purple-300 sm:px-2.5 sm:py-1 sm:text-xs">
                    {Math.round(smoothAmount * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.85}
                  step={0.01}
                  value={smoothAmount}
                  onChange={(event) => setSmoothAmount(Number(event.target.value))}
                  className="h-2.5 w-full cursor-pointer appearance-none rounded-full bg-gradient-to-r from-slate-700 to-slate-600 accent-purple-400 sm:h-2 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-purple-400 [&::-webkit-slider-thumb]:to-violet-400 [&::-webkit-slider-thumb]:shadow-lg sm:[&::-webkit-slider-thumb]:h-4 sm:[&::-webkit-slider-thumb]:w-4"
                />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between sm:mb-2.5">
                  <label className="text-xs font-medium text-slate-300 sm:text-sm">🌟 Glow & Warmth</label>
                  <span className="rounded-lg bg-cyan-500/20 px-2 py-0.5 text-[10px] font-bold text-cyan-300 sm:px-2.5 sm:py-1 sm:text-xs">
                    {Math.round(toneAmount * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.85}
                  step={0.01}
                  value={toneAmount}
                  onChange={(event) => setToneAmount(Number(event.target.value))}
                  className="h-2.5 w-full cursor-pointer appearance-none rounded-full bg-gradient-to-r from-slate-700 to-slate-600 accent-cyan-400 sm:h-2 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-cyan-400 [&::-webkit-slider-thumb]:to-blue-400 [&::-webkit-slider-thumb]:shadow-lg sm:[&::-webkit-slider-thumb]:h-4 sm:[&::-webkit-slider-thumb]:w-4"
                />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between sm:mb-2.5">
                  <label className="text-xs font-medium text-slate-300 sm:text-sm">💎 Clarity</label>
                  <span className="rounded-lg bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold text-emerald-300 sm:px-2.5 sm:py-1 sm:text-xs">
                    {Math.round(sharpenAmount * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.8}
                  step={0.01}
                  value={sharpenAmount}
                  onChange={(event) => setSharpenAmount(Number(event.target.value))}
                  className="h-2.5 w-full cursor-pointer appearance-none rounded-full bg-gradient-to-r from-slate-700 to-slate-600 accent-emerald-400 sm:h-2 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-emerald-400 [&::-webkit-slider-thumb]:to-green-400 [&::-webkit-slider-thumb]:shadow-lg sm:[&::-webkit-slider-thumb]:h-4 sm:[&::-webkit-slider-thumb]:w-4"
                />
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
