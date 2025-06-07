import { useCallback, useRef } from "react";
import { useEffect, useState } from "react";
import { ShaderMode } from "@/shared/webgpu/RenderPass";
import SmokeSimulation from "./SmokeSimulation";
import SimulationControls from "../components/SimulationControls";
import { usePersistedState } from "@/shared/utils/localStorage.utils";
import { PerformanceViewer } from "@/lib/performance";
import type { PerformanceMetrics } from "@/lib/performance";
import { usePerformanceToggle } from "@/lib/performance";
import { FunctionCallTracker, RAFEventProcessor } from "@/lib/performance";
import type { EventPerformanceMetrics } from "@/lib/performance";
import { SIMULATION_CONSTANTS } from "./constants";
import { type SmokeTextureID } from "./types";
import type { MouseEventData } from "@/lib/performance/types/eventMetrics";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/select";

const CANVAS_SCALE = 2;
const CANVAS_HEIGHT = SIMULATION_CONSTANTS.grid.size.height * CANVAS_SCALE;
const CANVAS_WIDTH = SIMULATION_CONSTANTS.grid.size.width * CANVAS_SCALE;

// Visualization mode options
const VISUALIZATION_MODES = [
  {
    value: ShaderMode.DENSITY,
    label: "Smoke Density",
    texture: "smokeDensity" as SmokeTextureID,
  },
  {
    value: ShaderMode.VELOCITY,
    label: "Velocity Field",
    texture: "velocity" as SmokeTextureID,
  },
  {
    value: ShaderMode.PRESSURE,
    label: "Pressure Field",
    texture: "pressure" as SmokeTextureID,
  },
];

function SmokeSimulationComponent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smokeSimulation = useRef<SmokeSimulation | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);

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
  const [selectedMode, setSelectedMode] = usePersistedState<ShaderMode>(
    "smoke-simulation-shader-mode",
    ShaderMode.DENSITY
  );
  const [selectedTexture, setSelectedTexture] =
    usePersistedState<SmokeTextureID>(
      "smoke-simulation-texture",
      "smokeDensity"
    );

  // Mouse interaction state - use refs to avoid triggering re-renders
  const isMouseDownRef = useRef(false);
  const previousMousePos = useRef<{ x: number; y: number } | null>(null);
  const lastInteractionTime = useRef<number>(0);

  // Cache canvas bounds to avoid frequent getBoundingClientRect calls
  const canvasBoundsRef = useRef<DOMRect | null>(null);
  const canvasBoundsUpdateTimeRef = useRef<number>(0);

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

  // Helper function to get cached canvas bounds
  const getCachedCanvasBounds = useCallback(() => {
    const now = performance.now();
    if (
      !canvasBoundsRef.current ||
      now - canvasBoundsUpdateTimeRef.current > 100
    ) {
      if (canvasRef.current) {
        canvasBoundsRef.current = canvasRef.current.getBoundingClientRect();
        canvasBoundsUpdateTimeRef.current = now;
      }
    }
    return canvasBoundsRef.current;
  }, []);

  // Helper function to convert canvas coordinates to grid coordinates
  const canvasToGridCoords = useCallback(
    (clientX: number, clientY: number) => {
      const rect = getCachedCanvasBounds();
      if (!rect) return null;

      const canvasX = clientX - rect.left;
      const canvasY = clientY - rect.top;

      // Scale to grid coordinates
      const gridX =
        (canvasX / rect.width) * SIMULATION_CONSTANTS.grid.size.width;
      // Note: Y coordinate should NOT be flipped since both canvas and WebGPU use top-left origin
      const gridY =
        (canvasY / rect.height) * SIMULATION_CONSTANTS.grid.size.height;

      return { x: gridX, y: gridY };
    },
    [getCachedCanvasBounds]
  );

  // Batched mouse event handler
  const handleBatchedMouseEvents = useCallback(
    (events: MouseEventData[]) => {
      // Track raw events for performance monitoring
      mouseMoveTracker.current.recordCall();

      if (!isInitialized || !smokeSimulation.current || events.length === 0)
        return;

      // Process the batch of events
      for (const event of events) {
        const coords = canvasToGridCoords(event.clientX, event.clientY);
        if (!coords) continue;

        if (event.type === "mousedown") {
          isMouseDownRef.current = true;
          smokeSimulation.current.addSmoke(coords.x, coords.y);
          previousMousePos.current = coords;
          lastInteractionTime.current = event.timestamp;
        } else if (event.type === "mousemove" && isMouseDownRef.current) {
          smokeSimulation.current.addSmoke(coords.x, coords.y);

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
        } else if (event.type === "mouseup") {
          isMouseDownRef.current = false;
          previousMousePos.current = null;
        }
      }
    },
    [isInitialized, canvasToGridCoords]
  );

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
  }, [handleBatchedMouseEvents]);

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

  // Cleanup effect - destroy simulation when component unmounts
  useEffect(() => {
    return () => {
      console.log("Component cleanup - destroying simulation");
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
      const simulation = new SmokeSimulation();
      smokeSimulation.current = simulation;

      try {
        await simulation.initialize(canvasRef);
        if (smokeSimulation.current === simulation) {
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
  }, []);

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

  const handleStep = useCallback(() => {
    if (smokeSimulation.current && isInitialized) {
      try {
        smokeSimulation.current.step({
          shaderMode: selectedMode,
          texture: selectedTexture,
        });
      } catch (error) {
        console.error("Failed to step simulation:", error);
      }
    }
  }, [isInitialized, selectedMode, selectedTexture]);

  const handleRestart = useCallback(async () => {
    if (!smokeSimulation.current || !isInitialized) {
      return;
    }

    try {
      smokeSimulation.current.reset(selectedMode, selectedTexture);
    } catch (error) {
      console.error("Failed to restart simulation:", error);
    }
  }, [isInitialized, selectedMode, selectedTexture]);

  const handleModeChange = useCallback(
    (value: string) => {
      const mode = parseInt(value) as ShaderMode;
      const modeOption = VISUALIZATION_MODES.find((m) => m.value === mode);
      if (modeOption) {
        setSelectedMode(mode);
        setSelectedTexture(modeOption.texture);
      }
    },
    [setSelectedMode, setSelectedTexture]
  );

  // Custom visualization mode dropdown
  const visualizationControl = (
    <div className="w-full">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Visualization Mode
      </label>
      <Select
        value={selectedMode.toString()}
        onValueChange={handleModeChange}
        disabled={!isInitialized}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select visualization mode" />
        </SelectTrigger>
        <SelectContent>
          {VISUALIZATION_MODES.map((mode) => (
            <SelectItem key={mode.value} value={mode.value.toString()}>
              {mode.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );

  return (
    <div>
      <PerformanceViewer
        metrics={performanceMetrics}
        eventMetrics={eventPerformanceMetrics}
        isVisible={isPerformanceVisible}
      />

      <div className="max-w-6xl mx-auto flex flex-col items-center gap-6">
        {/* Title and Description */}
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            Smoke Simulation
          </h2>
          <p className="text-sm text-gray-600 max-w-md">
            Click and drag on the simulation to add smoke and velocity. Click to
            add smoke, drag to create fluid motion.
          </p>
        </div>

        {initError && (
          <div className="text-red-500 text-sm max-w-md text-center bg-red-50 p-3 rounded-lg border border-red-200">
            <strong>Error:</strong> {initError}
          </div>
        )}

        {/* Main Content Area - Canvas and Controls */}
        <div className="flex flex-col md:flex-row items-center md:items-start gap-6 w-full justify-center">
          {/* Canvas */}
          <div className="flex-shrink-0">
            <canvas
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              ref={canvasRef}
              className="border border-gray-300 rounded-lg shadow-lg cursor-crosshair"
            />
          </div>

          {/* Controls */}
          <SimulationControls
            isInitialized={isInitialized}
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            onStep={handleStep}
            onRestart={handleRestart}
            title="Smoke Simulation"
            customControls={visualizationControl}
          />
        </div>
      </div>
    </div>
  );
}

export default SmokeSimulationComponent;
