import { useCallback, useRef } from "react";
import { useEffect, useState } from "react";
import { ShaderMode } from "@/shared/webgpu/RenderPass";
import SmokeSimulation from "@/lib/simulation/core/SmokeSimulation";
import { PerformanceViewer } from "@/lib/performance";
import type { PerformanceMetrics } from "@/lib/performance";
import { usePerformanceToggle } from "@/lib/performance";
import { FunctionCallTracker, RAFEventProcessor } from "@/lib/performance";
import type { EventPerformanceMetrics } from "@/lib/performance";
import { type SmokeTextureID } from "@/lib/simulation/core/types";
import type { MouseEventData } from "@/lib/performance/types/eventMetrics";
import { useAudioVisualization } from "@/lib/audio-visualization";
import { usePersistedState } from "@/shared/utils/localStorage.utils";
import {
  CanvasCoordinateConverter,
  isMouseNearBottom,
  clearLavaLampIntervals,
  scheduleLavaLampPill as scheduleLavaLampPillUtil,
  type LavaLampDependencies,
} from "@/lib/simulation/utils";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/shared/ui/dialog";
import { Button } from "@/shared/ui/button";
import { ColorPicker } from "./ColorPicker";
import { COLOR_PRESETS } from "../core/app.constants";

function SmokeSimulationComponent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smokeSimulation = useRef<SmokeSimulation | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [showWelcomeModal, setShowWelcomeModal] = useState(true);

  // Color picker state
  const [selectedColorIndex, setSelectedColorIndex] = usePersistedState(
    "fluxlab-color-index",
    0
  );
  const [showColorPicker, setShowColorPicker] = useState(false);

  // Lava lamp mode state
  const [lavaLampMode, setLavaLampMode] = useState(false);
  const lavaLampIntervalsRef = useRef<Set<number>>(new Set());

  // Window focus state for pausing/resuming lava lamp
  const [isWindowFocused, setIsWindowFocused] = useState(true);

  // Cursor visibility state
  const [isCursorVisible, setIsCursorVisible] = useState(true);
  const cursorTimeoutRef = useRef<number | null>(null);

  // Canvas dimensions state for viewport sizing
  const [canvasDimensions, setCanvasDimensions] = useState({
    width: typeof window !== "undefined" ? window.innerWidth : 800,
    height: typeof window !== "undefined" ? window.innerHeight : 600,
  });

  // Use refs for performance metrics to avoid React re-renders during animation
  const performanceMetricsRef = useRef<PerformanceMetrics>({
    js: 0,
    fps: 0,
    timestep: 0,
    gpu: 0,
    gpuSupported: false,
  });
  const [performanceMetrics, setPerformanceMetrics] =
    useState<PerformanceMetrics>(performanceMetricsRef.current);

  const isPerformanceVisible = usePerformanceToggle();
  const selectedMode = ShaderMode.DENSITY;
  const selectedTexture: SmokeTextureID = "smokeDensity";

  // Mouse interaction state - use refs to avoid triggering re-renders
  const isMouseDownRef = useRef(false);
  const previousMousePos = useRef<{ x: number; y: number } | null>(null);
  const lastInteractionTime = useRef<number>(0);

  // Canvas coordinate converter for efficient coordinate transformations
  const coordinateConverter = useRef(new CanvasCoordinateConverter());

  // Performance tracking
  const mouseMoveTracker = useRef(new FunctionCallTracker("Mouse Move Events"));
  const eventPerformanceMetricsRef = useRef<EventPerformanceMetrics>({
    mouseMoveCallsPerSecond: 0,
  });
  const [eventPerformanceMetrics, setEventPerformanceMetrics] =
    useState<EventPerformanceMetrics>(eventPerformanceMetricsRef.current);

  // RAF Event Processor for batching mouse events
  const rafEventProcessor = useRef<RAFEventProcessor<MouseEventData> | null>(
    null
  );

  // Audio visualization hook
  const audioVisualization = useAudioVisualization();
  const audioInitializedRef = useRef(false);

  // Handle color selection
  const handleColorSelect = useCallback((colorIndex: number) => {
    setSelectedColorIndex(colorIndex);
    if (smokeSimulation.current) {
      smokeSimulation.current.setSmokeColor(COLOR_PRESETS[colorIndex].color);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle lava lamp mode toggle
  const handleLavaLampModeChange = useCallback((enabled: boolean) => {
    setLavaLampMode(enabled);

    // Clear existing intervals when disabling
    if (!enabled) {
      clearLavaLampIntervals(lavaLampIntervalsRef.current);
    }
  }, []);

  // Handle mouse activity
  const handleMouseActivity = useCallback(
    (clientY: number) => {
      if (!showWelcomeModal && isMouseNearBottom(clientY)) {
        setShowColorPicker(true);
      }

      // Show cursor and reset hide timer
      if (!isCursorVisible) {
        setIsCursorVisible(true);
      }

      // Clear existing timeout
      if (cursorTimeoutRef.current) {
        clearTimeout(cursorTimeoutRef.current);
      }

      // Set new timeout to hide cursor after 3 seconds
      cursorTimeoutRef.current = setTimeout(() => {
        setIsCursorVisible(false);
      }, 3000);
    },
    [showWelcomeModal, isCursorVisible]
  );

  // Handle fullscreen functionality
  const enterFullscreen = useCallback(async () => {
    try {
      if (document.documentElement.requestFullscreen) {
        await document.documentElement.requestFullscreen();
      }
    } catch (error) {
      console.log("Fullscreen not supported or denied:", error);
      // Continue anyway - fullscreen is optional
    }
  }, []);

  // Handle "Enter" button click
  const handleEnter = useCallback(async () => {
    setShowWelcomeModal(false);
    await enterFullscreen();
  }, [enterFullscreen]);

  // Handle viewport resize
  useEffect(() => {
    const handleResize = () => {
      setCanvasDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
      // Clear cached canvas bounds when resizing
      coordinateConverter.current.clearCache();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Handle window focus/blur for pausing/resuming lava lamp
  useEffect(() => {
    const handleFocus = () => setIsWindowFocused(true);
    const handleBlur = () => setIsWindowFocused(false);

    window.addEventListener("focus", handleFocus);
    window.addEventListener("blur", handleBlur);

    return () => {
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  // Get lava lamp dependencies for utility functions
  const getLavaLampDependencies =
    useCallback((): LavaLampDependencies | null => {
      if (!smokeSimulation.current || !canvasRef.current) return null;

      const canvasBounds = canvasRef.current.getBoundingClientRect();

      return {
        simulation: smokeSimulation.current,
        audioVisualization,
        canvasBounds,
        isLavaLampMode: lavaLampMode,
        intervalTracker: lavaLampIntervalsRef.current,
        isWindowFocused: isWindowFocused,
      };
    }, [audioVisualization, lavaLampMode, isWindowFocused]);

  // Wrapper for scheduling lava lamp pills
  const scheduleLavaLampPill = useCallback(() => {
    const dependencies = getLavaLampDependencies();
    if (!dependencies) return;

    scheduleLavaLampPillUtil(dependencies, scheduleLavaLampPill);
  }, [getLavaLampDependencies]);

  // Batched mouse event handler
  const handleBatchedMouseEvents = useCallback(
    (events: MouseEventData[]) => {
      // Track raw events for performance monitoring
      mouseMoveTracker.current.recordCall();

      if (!isInitialized || !smokeSimulation.current || events.length === 0)
        return;

      // Process the batch of events
      for (const event of events) {
        const coords = coordinateConverter.current.clientToGrid(
          event.clientX,
          event.clientY
        );
        if (!coords) continue;

        if (event.type === "mousedown") {
          isMouseDownRef.current = true;
          smokeSimulation.current.addSmoke(coords.x, coords.y);
          previousMousePos.current = coords;
          lastInteractionTime.current = event.timestamp;

          // Handle audio visualization
          audioVisualization.handleMouseDown(event.clientX, event.clientY);
        } else if (event.type === "mousemove") {
          // Show color picker on mouse activity near bottom
          handleMouseActivity(event.clientY);

          if (isMouseDownRef.current) {
            smokeSimulation.current.addSmoke(coords.x, coords.y);

            // Handle audio visualization
            audioVisualization.handleMouseMove(event.clientX, event.clientY);

            if (previousMousePos.current) {
              const deltaTime =
                (event.timestamp - lastInteractionTime.current) / 1000;

              if (deltaTime > 0) {
                const rawVelocityX =
                  (coords.x - previousMousePos.current.x) / deltaTime;
                const rawVelocityY =
                  (coords.y - previousMousePos.current.y) / deltaTime;

                const velocityScale = 0.02;
                const maxVelocity = 10;

                let velocityX = rawVelocityX * velocityScale;
                let velocityY = rawVelocityY * velocityScale;

                const velocityMagnitude = Math.sqrt(
                  velocityX * velocityX + velocityY * velocityY
                );
                if (velocityMagnitude > maxVelocity) {
                  const scale = maxVelocity / velocityMagnitude;
                  velocityX *= scale;
                  velocityY *= scale;
                }

                smokeSimulation.current.addVelocity(
                  coords.x,
                  coords.y,
                  velocityX,
                  velocityY
                );
              }

              previousMousePos.current = coords;
              lastInteractionTime.current = event.timestamp;
            }
          }
        } else if (event.type === "mouseup") {
          isMouseDownRef.current = false;
          previousMousePos.current = null;

          // Handle audio visualization
          audioVisualization.handleMouseUp(event.clientX, event.clientY);
        }
      }
    },
    [isInitialized, audioVisualization, handleMouseActivity]
  );

  // Initialize audio visualization and coordinate converter when simulation is ready
  useEffect(() => {
    if (
      isInitialized &&
      smokeSimulation.current &&
      canvasRef.current &&
      !audioInitializedRef.current
    ) {
      audioVisualization.initialize(canvasRef.current, smokeSimulation.current);
      coordinateConverter.current.initialize(
        canvasRef.current,
        smokeSimulation.current
      );
      audioInitializedRef.current = true;
    }
  }, [isInitialized, audioVisualization]);

  // Initialize RAF event processor and native event listeners to bypass React overhead
  useEffect(() => {
    rafEventProcessor.current = new RAFEventProcessor(handleBatchedMouseEvents);

    const canvas = canvasRef.current;
    if (!canvas) return;

    // Native event handlers (bypass React's synthetic event system)
    const handleNativeMouseDown = (event: MouseEvent) => {
      rafEventProcessor.current?.addEvent({
        clientX: event.clientX,
        clientY: event.clientY,
        type: "mousedown",
        timestamp: performance.now(),
      });
    };

    const handleNativeMouseMove = (event: MouseEvent) => {
      rafEventProcessor.current?.addEvent({
        clientX: event.clientX,
        clientY: event.clientY,
        type: "mousemove",
        timestamp: performance.now(),
      });
    };

    const handleNativeMouseUp = (event: MouseEvent) => {
      rafEventProcessor.current?.addEvent({
        clientX: event.clientX,
        clientY: event.clientY,
        type: "mouseup",
        timestamp: performance.now(),
      });
    };

    const handleNativeMouseLeave = () => {
      isMouseDownRef.current = false;
      previousMousePos.current = null;

      // Handle audio visualization cleanup
      audioVisualization.handleMouseLeave();
    };

    // Add native event listeners with passive option for better performance
    canvas.addEventListener("mousedown", handleNativeMouseDown);
    canvas.addEventListener("mousemove", handleNativeMouseMove, {
      passive: true,
    });
    canvas.addEventListener("mouseup", handleNativeMouseUp);
    canvas.addEventListener("mouseleave", handleNativeMouseLeave);

    // Cleanup
    return () => {
      rafEventProcessor.current?.destroy();

      if (canvas) {
        canvas.removeEventListener("mousedown", handleNativeMouseDown);
        canvas.removeEventListener("mousemove", handleNativeMouseMove);
        canvas.removeEventListener("mouseup", handleNativeMouseUp);
        canvas.removeEventListener("mouseleave", handleNativeMouseLeave);
      }
    };
  }, [handleBatchedMouseEvents, audioVisualization]);

  // Update performance stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      if (smokeSimulation.current) {
        setPerformanceMetrics(smokeSimulation.current.getPerformanceMetrics());
      }
      setEventPerformanceMetrics({
        mouseMoveCallsPerSecond: mouseMoveTracker.current.getCallsPerSecond(),
      });
    }, 100); // Update every 100ms

    return () => clearInterval(interval);
  }, []);

  // Lava lamp mode effect
  useEffect(() => {
    if (lavaLampMode && isInitialized) {
      // Clear any existing scheduling first
      clearLavaLampIntervals(lavaLampIntervalsRef.current);

      // Start new scheduling
      scheduleLavaLampPill();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lavaLampMode, isInitialized]);

  // Resume lava lamp when window regains focus
  useEffect(() => {
    if (lavaLampMode && isInitialized && isWindowFocused) {
      // Clear any existing scheduling first
      clearLavaLampIntervals(lavaLampIntervalsRef.current);
      scheduleLavaLampPill();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isWindowFocused]);

  // Cleanup effect - destroy simulation when component unmounts
  useEffect(() => {
    return () => {
      console.log("Component cleanup - destroying simulation");

      // Clear cursor timeout
      if (cursorTimeoutRef.current) {
        clearTimeout(cursorTimeoutRef.current);
      }

      // Clear all lava lamp intervals
      // eslint-disable-next-line react-hooks/exhaustive-deps
      clearLavaLampIntervals(lavaLampIntervalsRef.current);

      if (smokeSimulation.current) {
        smokeSimulation.current.destroy();
        smokeSimulation.current = null;
      }
      setIsInitialized(false);
    };
  }, []);

  // Initialize simulation
  useEffect(() => {
    const runSimulation = async () => {
      if (!canvasRef.current || smokeSimulation.current) {
        return;
      }

      console.log("Initializing Smoke Simulation");

      const simulationConfig = {
        grid: {
          size: {
            width: 512,
            height: Math.floor(
              512 * (canvasDimensions.height / canvasDimensions.width)
            ),
          },
          scale: 0.9,
        },
      };

      const simulation = new SmokeSimulation(simulationConfig);
      smokeSimulation.current = simulation;

      try {
        await simulation.initialize(canvasRef);
        if (smokeSimulation.current === simulation) {
          // Set initial color
          simulation.setSmokeColor(COLOR_PRESETS[selectedColorIndex].color);
          setIsInitialized(true);
          setInitError(null);
        }
      } catch (error) {
        console.error("Failed to initialize smoke simulation:", error);
        if (smokeSimulation.current === simulation) {
          simulation.destroy();
          smokeSimulation.current = null;
          setIsInitialized(false);
          setInitError(
            error instanceof Error ? error.message : "Unknown error"
          );
        }
      }
    };
    runSimulation();
  }, [canvasDimensions, selectedColorIndex]); // Add selectedColorIndex as dependency

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !isInitialized || !smokeSimulation.current) {
      return;
    }

    const animate = () => {
      if (smokeSimulation.current && isPlaying) {
        try {
          smokeSimulation.current.step({
            shaderMode: selectedMode,
            texture: selectedTexture,
          });

          animationFrameRef.current = requestAnimationFrame(animate);
        } catch (error) {
          console.error("Failed to step simulation:", error);
          setIsPlaying(false);
        }
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isPlaying, isInitialized, selectedMode, selectedTexture]);

  // Re-render when visualization mode changes (if not playing)
  useEffect(() => {
    if (isInitialized && !isPlaying && smokeSimulation.current) {
      try {
        smokeSimulation.current.step({
          renderOnly: true,
          shaderMode: selectedMode,
          texture: selectedTexture,
        });
      } catch (error) {
        console.error("Failed to update visualization:", error);
      }
    }
  }, [selectedMode, selectedTexture, isInitialized, isPlaying]);

  return (
    <>
      {/* Welcome Modal */}
      <Dialog open={showWelcomeModal} onOpenChange={() => {}}>
        <DialogContent
          hideCloseButton={true}
          className="max-w-lg bg-black/95 backdrop-blur-md border-neutral-800 text-neutral-100 shadow-2xl"
        >
          <DialogHeader className="text-center space-y-6">
            <DialogTitle
              className="text-center text-4xl font-normal tracking-wide text-neutral-100 flex items-center justify-center gap-3"
              style={{ fontFamily: "Baskervville, serif" }}
            >
              <img
                src="/flux-lab/Astroid_filled.svg"
                alt="FluxLab Icon"
                className="w-9 h-9 brightness-0 invert"
              />
              FluxLab
            </DialogTitle>
            <div className="space-y-4">
              <DialogDescription
                className="text-neutral-400 text-base font-light tracking-wide text-center"
                style={{ fontFamily: "Baskervville, serif" }}
              >
                Music from movement.
              </DialogDescription>
              <div className="w-16 h-px bg-neutral-600 mx-auto"></div>
              <DialogDescription
                className="text-neutral-400 text-base font-light tracking-wide text-center"
                style={{ fontFamily: "Baskervville, serif" }}
              >
                Click and drag to synthesize. Seek the spectrum in the depths
                below.
              </DialogDescription>
            </div>
          </DialogHeader>
          <DialogFooter className="mt-8">
            <Button
              onClick={handleEnter}
              variant="outline"
              className="w-full bg-transparent border-neutral-600 text-neutral-100 hover:bg-neutral-900 hover:border-neutral-500 py-3 text-lg font-light tracking-wider transition-all duration-500 hover:text-neutral-200"
              style={{
                fontFamily: "Baskervville, serif",
                letterSpacing: "0.1em",
                transition: "all 0.5s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.letterSpacing = "0.3em";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.letterSpacing = "0.1em";
              }}
            >
              Enter
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Main Simulation */}
      <div className="w-screen h-screen overflow-hidden relative">
        <PerformanceViewer
          metrics={performanceMetrics}
          eventMetrics={eventPerformanceMetrics}
          isVisible={isPerformanceVisible}
        />

        {/* Color Picker Component */}
        <ColorPicker
          selectedColorIndex={selectedColorIndex}
          onColorSelect={handleColorSelect}
          isVisible={showColorPicker}
          onVisibilityChange={setShowColorPicker}
          lavaLampMode={lavaLampMode}
          onLavaLampModeChange={handleLavaLampModeChange}
        />

        {initError && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 text-red-500 text-sm bg-red-50 p-3 rounded-lg border border-red-200">
            <strong>Error:</strong> {initError}
          </div>
        )}

        <canvas
          width={canvasDimensions.width}
          height={canvasDimensions.height}
          ref={canvasRef}
          className="block"
          style={{
            width: "100vw",
            height: "100vh",
            cursor: isCursorVisible ? "crosshair" : "none",
          }}
        />
      </div>
    </>
  );
}

export default SmokeSimulationComponent;
