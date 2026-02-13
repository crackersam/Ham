"use client";

import { useEffect, useRef, useState } from "react";

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

// Jeeliz FaceFilter type definitions
declare global {
  interface Window {
    JEELIZFACEFILTER?: {
      init: (params: {
        canvasId?: string;
        canvas?: HTMLCanvasElement;
        NNCPath?: string;
        callbackReady?: (errCode: string, spec: { GL: WebGLRenderingContext; canvasElement: HTMLCanvasElement }) => void;
        callbackTrack?: (detectState: { detected: number; x: number; y: number; s: number; rx: number; ry: number; rz: number }) => void;
      }) => void;
      get_rotationStabilized: () => [number, number, number];
      get_positionScale: () => [number, number, number];
      is_detected: () => boolean;
    };
  }
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

export default function MakeupStudioJeeliz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const scriptLoadedRef = useRef(false);
  const jeelizInitializedRef = useRef(false);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [status, setStatus] = useState("Loading Jeeliz FaceFilter...");
  const [isRunning, setIsRunning] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [activeStyleId, setActiveStyleId] = useState(STYLES[0].id);
  const [masterIntensity, setMasterIntensity] = useState(0.95);

  const activeStyleRef = useRef(STYLES[0]);
  const masterIntensityRef = useRef(0.95);

  useEffect(() => {
    activeStyleRef.current = STYLES.find((s) => s.id === activeStyleId) ?? STYLES[0];
  }, [activeStyleId]);

  useEffect(() => {
    masterIntensityRef.current = masterIntensity;
  }, [masterIntensity]);

  useEffect(() => {
    let disposed = false;

    const loadJeelizScript = () => {
      return new Promise<void>((resolve, reject) => {
        if (scriptLoadedRef.current || document.querySelector('script[src*="jeelizFaceFilter"]')) {
          resolve();
          return;
        }

        const script = document.createElement("script");
        // Try the direct GitHub CDN URL
        script.src = "https://cdn.jsdelivr.net/gh/jeeliz/jeelizFaceFilter@master/dist/jeelizFaceFilter.min.js";
        script.async = true;
        script.onload = () => {
          console.log("Jeeliz script loaded successfully");
          console.log("JEELIZFACEFILTER available:", !!window.JEELIZFACEFILTER);
          scriptLoadedRef.current = true;
          resolve();
        };
        script.onerror = (e) => {
          console.error("Failed to load Jeeliz script:", e);
          reject(new Error("Failed to load Jeeliz FaceFilter script"));
        };
        document.head.appendChild(script);
      });
    };

    const initJeeliz = async () => {
      try {
        await loadJeelizScript();

        if (disposed) return;

        const canvas = canvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        
        if (!canvas || !overlayCanvas) {
          console.error("Canvas elements not found");
          return;
        }

        setStatus("Initializing Jeeliz FaceFilter...");

        if (!window.JEELIZFACEFILTER) {
          throw new Error("Jeeliz FaceFilter not loaded");
        }

        console.log("Starting Jeeliz initialization...");

        // Initialize Jeeliz FaceFilter
        window.JEELIZFACEFILTER.init({
          canvas: canvas,
          NNCPath: "https://cdn.jsdelivr.net/gh/jeeliz/jeelizFaceFilter@master/dist/",
          followZRot: true,
          maxFacesDetected: 1,
          callbackReady: (errCode, spec) => {
            if (disposed) return;

            if (errCode) {
              console.error("Jeeliz initialization error:", errCode);
              let errorMessage = "Unknown error";
              if (errCode === "WEBCAM_UNAVAILABLE") {
                errorMessage = "Camera unavailable or permission denied";
              } else if (errCode === "GL_INCOMPATIBLE") {
                errorMessage = "WebGL not supported";
              } else if (errCode === "ALREADY_INITIALIZED") {
                errorMessage = "Jeeliz already initialized";
              } else if (errCode === "NO_CANVASID") {
                errorMessage = "Canvas not found";
              }
              setStatus(`Error: ${errorMessage}`);
              setIsRunning(false);
              return;
            }

            console.log("Jeeliz initialized successfully", spec);
            glRef.current = spec.GL;
            jeelizInitializedRef.current = true;
            setStatus("Live: Jeeliz face tracking");
            setIsRunning(true);

            // Set overlay canvas size to match video canvas
            const width = spec.canvasElement.width;
            const height = spec.canvasElement.height;
            overlayCanvas.width = width;
            overlayCanvas.height = height;

            console.log("Canvas dimensions:", width, "x", height);

            // Start render loop
            startRenderLoop();
          },
          callbackTrack: (detectState) => {
            if (disposed) return;
            setFaceDetected(detectState.detected > 0.5);
          },
        });
      } catch (error) {
        console.error("Jeeliz setup error:", error);
        if (!disposed) {
          setStatus(error instanceof Error ? error.message : "Failed to initialize Jeeliz");
          setIsRunning(false);
        }
      }
    };

    const drawMakeup = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      ctx.clearRect(0, 0, width, height);

      if (!window.JEELIZFACEFILTER || !window.JEELIZFACEFILTER.is_detected()) {
        return;
      }

      const style = activeStyleRef.current;
      const intensity = masterIntensityRef.current;

      // Get face position and scale
      const [x, y, s] = window.JEELIZFACEFILTER.get_positionScale();
      const [rx, ry, rz] = window.JEELIZFACEFILTER.get_rotationStabilized();

      // Convert normalized coordinates to canvas coordinates
      const centerX = ((x + 1) / 2) * width;
      const centerY = ((1 - y) / 2) * height;
      const scale = s * width * 0.5;

      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(rz);

      // Draw lips
      const lipY = scale * 0.3;
      const lipWidth = scale * 0.35;
      const lipHeight = scale * 0.15;
      ctx.fillStyle = hexToRgba(style.lipstick, style.lipstickOpacity * intensity * 0.8);
      ctx.filter = "blur(4px)";
      ctx.beginPath();
      ctx.ellipse(0, lipY, lipWidth, lipHeight, 0, 0, Math.PI * 2);
      ctx.fill();

      // Draw blush (left)
      const blushY = scale * 0.05;
      const blushRadius = scale * 0.2;
      const blushGradientLeft = ctx.createRadialGradient(
        -scale * 0.35,
        blushY,
        0,
        -scale * 0.35,
        blushY,
        blushRadius
      );
      blushGradientLeft.addColorStop(0, hexToRgba(style.blush, style.blushOpacity * intensity * 0.5));
      blushGradientLeft.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = blushGradientLeft;
      ctx.filter = "blur(12px)";
      ctx.beginPath();
      ctx.arc(-scale * 0.35, blushY, blushRadius, 0, Math.PI * 2);
      ctx.fill();

      // Draw blush (right)
      const blushGradientRight = ctx.createRadialGradient(
        scale * 0.35,
        blushY,
        0,
        scale * 0.35,
        blushY,
        blushRadius
      );
      blushGradientRight.addColorStop(0, hexToRgba(style.blush, style.blushOpacity * intensity * 0.5));
      blushGradientRight.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = blushGradientRight;
      ctx.filter = "blur(12px)";
      ctx.beginPath();
      ctx.arc(scale * 0.35, blushY, blushRadius, 0, Math.PI * 2);
      ctx.fill();

      // Draw eyeshadow (left)
      const eyeY = -scale * 0.15;
      const eyeWidth = scale * 0.15;
      const eyeHeight = scale * 0.12;
      ctx.fillStyle = hexToRgba(style.eyeshadow, style.eyeshadowOpacity * intensity * 0.6);
      ctx.filter = "blur(8px)";
      ctx.beginPath();
      ctx.ellipse(-scale * 0.25, eyeY, eyeWidth, eyeHeight, 0, 0, Math.PI * 2);
      ctx.fill();

      // Draw eyeshadow (right)
      ctx.beginPath();
      ctx.ellipse(scale * 0.25, eyeY, eyeWidth, eyeHeight, 0, 0, Math.PI * 2);
      ctx.fill();

      // Draw eyeliner (left)
      ctx.strokeStyle = hexToRgba(style.liner, style.linerOpacity * intensity * 0.8);
      ctx.lineWidth = scale * 0.012;
      ctx.lineCap = "round";
      ctx.filter = "blur(1px)";
      ctx.beginPath();
      ctx.moveTo(-scale * 0.35, eyeY);
      ctx.lineTo(-scale * 0.15, eyeY);
      ctx.stroke();

      // Draw eyeliner (right)
      ctx.beginPath();
      ctx.moveTo(scale * 0.15, eyeY);
      ctx.lineTo(scale * 0.35, eyeY);
      ctx.stroke();

      ctx.restore();
    };

    const startRenderLoop = () => {
      const overlayCanvas = overlayCanvasRef.current;
      if (!overlayCanvas) return;

      const ctx = overlayCanvas.getContext("2d");
      if (!ctx) return;

      const render = () => {
        if (disposed) return;

        drawMakeup(ctx, overlayCanvas.width, overlayCanvas.height);
        animationFrameRef.current = requestAnimationFrame(render);
      };

      render();
    };

    initJeeliz();

    return () => {
      disposed = true;
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      jeelizInitializedRef.current = false;
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950/30 to-slate-900">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              <span className="bg-gradient-to-r from-fuchsia-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
                Artist Makeup Studio
              </span>
              <span className="ml-3 rounded-lg bg-emerald-500/20 px-2 py-1 text-xs font-medium text-emerald-300">
                Jeeliz
              </span>
            </h1>
            <p className="mt-1 text-sm text-slate-400">Powered by Jeeliz FaceFilter</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${isRunning ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
              <span className="text-sm font-medium text-slate-300">
                {isRunning ? "Live" : "Loading"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl gap-8 px-6 py-8 lg:grid-cols-[1.5fr_1fr]">
        {/* Video Section */}
        <section className="space-y-4">
          <div className="overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-4 shadow-2xl backdrop-blur-xl">
            <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black shadow-inner" style={{ aspectRatio: '16/9' }}>
              <canvas 
                ref={canvasRef} 
                className="absolute inset-0 h-full w-full"
                width={1280}
                height={720}
              />
              <canvas 
                ref={overlayCanvasRef} 
                className="pointer-events-none absolute inset-0 h-full w-full"
              />
              
              {/* Overlay UI */}
              <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between p-5">
                <div className="rounded-2xl border border-white/20 bg-black/40 px-4 py-2 backdrop-blur-md">
                  <p className="text-xs font-medium text-white/90">Live Preview</p>
                </div>
                <div className="flex items-center gap-3">
                  {faceDetected && (
                    <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/20 px-4 py-2 backdrop-blur-md">
                      <p className="text-xs font-medium text-emerald-300">Face Detected</p>
                    </div>
                  )}
                  <div className="rounded-2xl border border-white/20 bg-black/40 px-4 py-2 backdrop-blur-md">
                    <p className="text-xs font-medium text-white/90">{activeStyleRef.current.name}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Status Bar */}
          <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-gradient-to-r from-slate-900/80 to-slate-800/80 px-5 py-3 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-fuchsia-500 to-pink-500">
                <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-white">{status}</p>
                <p className="text-xs text-slate-400">Jeeliz Face Tracking Active</p>
              </div>
            </div>
          </div>
        </section>

        {/* Controls Panel */}
        <aside className="space-y-6">
          {/* Makeup Styles */}
          <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-6 shadow-2xl backdrop-blur-xl">
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Makeup Styles</h2>
              <div className="rounded-lg bg-fuchsia-500/20 px-3 py-1 text-xs font-medium text-fuchsia-300">
                {STYLES.length} Looks
              </div>
            </div>

            <div className="space-y-2.5">
              {STYLES.map((style) => (
                <button
                  key={style.id}
                  type="button"
                  onClick={() => setActiveStyleId(style.id)}
                  className={`group relative w-full overflow-hidden rounded-xl border p-4 text-left transition-all duration-300 ${
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
                      <div className="flex items-center gap-2">
                        <p className="font-semibold text-white">{style.name}</p>
                        {activeStyleId === style.id && (
                          <svg className="h-4 w-4 text-fuchsia-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                        )}
                      </div>
                      <div className="mt-2.5 flex gap-2">
                        {[
                          { color: style.lipstick, label: "Lip" },
                          { color: style.eyeshadow, label: "Eye" },
                          { color: style.blush, label: "Blush" },
                          { color: style.liner, label: "Liner" }
                        ].map((swatch) => (
                          <div key={`${style.id}-${swatch.color}`} className="flex flex-col items-center gap-1">
                            <span
                              className="h-5 w-5 rounded-full border-2 border-white/40 shadow-lg ring-2 ring-black/20"
                              style={{ backgroundColor: swatch.color }}
                            />
                            <span className="text-[9px] font-medium text-slate-400">{swatch.label}</span>
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
          <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-6 shadow-2xl backdrop-blur-xl">
            <h2 className="mb-5 text-lg font-semibold text-white">Style Intensity</h2>
            
            <div className="space-y-6">
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300">
                    <svg className="h-4 w-4 text-fuchsia-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                    </svg>
                    Makeup Intensity
                  </label>
                  <span className="rounded-lg bg-fuchsia-500/20 px-2.5 py-1 text-xs font-bold text-fuchsia-300">
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
                  className="h-2 w-full cursor-pointer appearance-none rounded-full bg-gradient-to-r from-slate-700 to-slate-600 accent-fuchsia-400 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-fuchsia-400 [&::-webkit-slider-thumb]:to-pink-400 [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:ring-2 [&::-webkit-slider-thumb]:ring-fuchsia-300/50"
                />
              </div>
            </div>
          </div>

          {/* Info Box */}
          <div className="rounded-3xl border border-emerald-400/20 bg-gradient-to-br from-emerald-900/20 to-slate-950/90 p-6 shadow-2xl backdrop-blur-xl">
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/20">
                <svg className="h-4 w-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">About Jeeliz</h3>
                <p className="mt-1 text-xs text-slate-400">
                  Jeeliz FaceFilter provides lightweight, real-time face tracking using WebGL. 
                  It's optimized for performance and works well on mobile devices.
                </p>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
