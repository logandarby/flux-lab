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
import type { SmokeTextureExports } from "./SmokeSimulation";
import type { MouseEventData } from "@/lib/performance/types/eventMetrics";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/select";
import {
  AudioEngine,
  type Note,
  type NoteVelocity,
} from "@/lib/audio-engine/core/AudioEngine";

const CANVAS_SCALE = 2;
const CANVAS_HEIGHT = SIMULATION_CONSTANTS.grid.size.height * CANVAS_SCALE;
const CANVAS_WIDTH = SIMULATION_CONSTANTS.grid.size.width * CANVAS_SCALE;

// Audio constants
const AUDIO_SAMPLE_INTERVAL_MS = 100; // Sample audio every 100ms
const PENTATONIC_SCALE = [
  // Three-octave major pentatonic scale in Hz (low to high)
  261.63,
  293.66,
  329.63,
  392.0,
  440.0, // C4, D4, E4, G4, A4
  523.25,
  587.33,
  659.25,
  783.99,
  880.0, // C5, D5, E5, G5, A5
  1046.5,
  1174.66,
  1318.51,
  1567.98,
  1760.0, // C6, D6, E6, G6, A6
];

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
  const isPlayingRef = useRef<boolean>(true);
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

  // Audio engine state
  const audioEngine = useRef<AudioEngine | null>(null);
  const audioInitialized = useRef<boolean>(false);
  const currentMousePos = useRef<{ x: number; y: number } | null>(null);
  const lastPlayedNote = useRef<Note | null>(null);
  const sampledTextureData = useRef<SmokeTextureExports | null>(null);
  const samplingIntervalRef = useRef<number | null>(null);

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

  // Helper function to compare if two notes are the same
  const areNotesEqual = useCallback(
    (note1: Note | null, note2: Note): boolean => {
      if (!note1) return false;
      return note1.pitch === note2.pitch && note1.velocity === note2.velocity;
    },
    []
  );

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

  // Function to sample textures from simulation (called on interval when mouse is down)
  const sampleTextures = useCallback(async () => {
    if (
      !audioInitialized.current ||
      !smokeSimulation.current ||
      !isPlayingRef.current
    ) {
      return;
    }

    try {
      // Sample textures from simulation and store for use
      const textureData = await smokeSimulation.current.sampleTextures(4); // Downsample for performance
      sampledTextureData.current = textureData;
    } catch (error) {
      console.warn("Failed to sample textures:", error);
    }
  }, []);

  // Function to play note based on mouse position (called on every mousemove)
  const playNoteAtPosition = useCallback(
    async (clientX: number, clientY: number) => {
      if (
        !audioInitialized.current ||
        !sampledTextureData.current ||
        !isPlayingRef.current
      ) {
        return;
      }

      try {
        // Get normalized coordinates from current mouse position
        const rect = getCachedCanvasBounds();
        if (!rect) return;

        const normalizedX = (clientX - rect.left) / rect.width;
        // Flip Y coordinate so top = low pitch, bottom = high pitch
        const normalizedY = 1.0 - (clientY - rect.top) / rect.height;

        // Ensure coordinates are in valid range
        if (
          normalizedX < 0 ||
          normalizedX > 1 ||
          normalizedY < 0 ||
          normalizedY > 1
        ) {
          return;
        }

        // Sample density at normalized coordinates
        const density = sampledTextureData.current.smokeDensity.getAtNormalized(
          normalizedX,
          normalizedY
        );

        // Map normalized Y to pentatonic scale (0 = lowest pitch, 1 = highest pitch now)
        const scaleIndex = Math.floor(
          normalizedY * (PENTATONIC_SCALE.length - 1)
        );
        const pitch = PENTATONIC_SCALE[scaleIndex];

        // Map density to velocity (0-1 range, with minimum threshold for audibility)
        const velocity = Math.max(
          0.5,
          Math.min(1.0, density * 2)
        ) as NoteVelocity;

        // Create note and play it
        const note: Note = {
          pitch: pitch,
          velocity: velocity,
        };

        // Only play if the note is different from the previous one
        if (!areNotesEqual(lastPlayedNote.current, note)) {
          await audioEngine.current?.playNote(note);
          lastPlayedNote.current = note;
        }
      } catch (error) {
        console.warn("Failed to play note:", error);
      }
    },
    [getCachedCanvasBounds, areNotesEqual]
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

          // Start texture sampling interval when mouse is pressed
          if (!samplingIntervalRef.current) {
            sampleTextures(); // Sample immediately
            samplingIntervalRef.current = window.setInterval(
              sampleTextures,
              AUDIO_SAMPLE_INTERVAL_MS
            );
          }

          // Play initial note on click
          playNoteAtPosition(event.clientX, event.clientY);
        } else if (event.type === "mousemove") {
          // Always track mouse position
          currentMousePos.current = { x: event.clientX, y: event.clientY };

          if (isMouseDownRef.current) {
            smokeSimulation.current.addSmoke(coords.x, coords.y);

            // Play note on every mousemove when dragging
            playNoteAtPosition(event.clientX, event.clientY);

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
          // Reset last played note so clicking the same spot again will play the note
          lastPlayedNote.current = null;
          // Stop texture sampling interval when mouse is released
          if (samplingIntervalRef.current) {
            clearInterval(samplingIntervalRef.current);
            samplingIntervalRef.current = null;
          }
        }
      }
    },
    [isInitialized, canvasToGridCoords, sampleTextures, playNoteAtPosition]
  );

  // Initialize audio engine on user gesture
  useEffect(() => {
    if (audioInitialized.current || !canvasRef.current) return;

    const handleUserGesture = async () => {
      if (!audioEngine.current) {
        try {
          audioEngine.current = new AudioEngine();
          await audioEngine.current.initialize();
          audioInitialized.current = true;

          console.log("Audio engine initialized successfully");
        } catch (error) {
          console.warn("Failed to initialize audio engine:", error);
        }
      }

      // Remove listeners immediately after initialization
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.removeEventListener("click", handleUserGesture);
        canvas.removeEventListener("mousedown", handleUserGesture);
        canvas.removeEventListener("keydown", handleUserGesture);
        document.removeEventListener("keydown", handleUserGesture);
      }
    };

    // Add multiple user gesture listeners to ensure we catch user interaction
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.addEventListener("click", handleUserGesture, { once: true });
      canvas.addEventListener("mousedown", handleUserGesture, { once: true });
      canvas.addEventListener("keydown", handleUserGesture, { once: true });
      document.addEventListener("keydown", handleUserGesture, { once: true });
    }

    // Cleanup if component unmounts before initialization
    return () => {
      if (canvas) {
        canvas.removeEventListener("click", handleUserGesture);
        canvas.removeEventListener("mousedown", handleUserGesture);
        canvas.removeEventListener("keydown", handleUserGesture);
        document.removeEventListener("keydown", handleUserGesture);
      }
    };
  }, [sampleTextures]);

  // Keep isPlayingRef in sync with isPlaying state
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

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
      // Reset last played note when mouse leaves canvas
      lastPlayedNote.current = null;
      // Stop texture sampling interval when mouse leaves canvas
      if (samplingIntervalRef.current) {
        clearInterval(samplingIntervalRef.current);
        samplingIntervalRef.current = null;
      }
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
      // Cleanup audio engine and sampling interval
      if (samplingIntervalRef.current) {
        clearInterval(samplingIntervalRef.current);
        samplingIntervalRef.current = null;
      }
      if (audioEngine.current) {
        audioEngine.current = null;
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
