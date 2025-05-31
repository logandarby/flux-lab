import { useCallback, useRef } from "react";
import { useEffect, useState } from "react";
import { ShaderMode } from "../utils/RenderPass";
import SimulationControls from "./ui/SimulationControls";
import SmokeSimulation, {
  GRID_SIZE,
  type SmokeTextureID,
} from "../SmokeSimulation";
import { usePersistedState } from "../utils/localStorage.utils";

const CANVAS_HEIGHT = 512;
const CANVAS_WIDTH = CANVAS_HEIGHT;

function SmokeSimulationComponent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smokeSimulation = useRef<SmokeSimulation | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = usePersistedState<ShaderMode>(
    "smoke-simulation-shader-mode",
    ShaderMode.DENSITY
  );
  const [selectedTexture, setSelectedTexture] =
    usePersistedState<SmokeTextureID>(
      "smoke-simulation-texture",
      "smokeDensity"
    );

  // Mouse interaction state
  const [isMouseDown, setIsMouseDown] = useState(false);
  const previousMousePos = useRef<{ x: number; y: number } | null>(null);
  const lastInteractionTime = useRef<number>(0);

  useEffect(() => {
    const runSimulation = async () => {
      if (!canvasRef.current || !!smokeSimulation.current || isInitialized) {
        return;
      }
      console.log("Initializing Smoke Simulation");
      smokeSimulation.current = new SmokeSimulation();
      try {
        await smokeSimulation.current.initialize(canvasRef);
        setIsInitialized(true);
        setInitError(null);
      } catch (error) {
        console.error("Failed to initialize smoke simulation:", error);
        smokeSimulation.current = null;
        setIsInitialized(false);
        setInitError(error instanceof Error ? error.message : "Unknown error");
      }
    };
    runSimulation();
  }, [isInitialized]);

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

  // Helper function to convert canvas coordinates to grid coordinates
  const canvasToGridCoords = useCallback((clientX: number, clientY: number) => {
    if (!canvasRef.current) return null;

    const rect = canvasRef.current.getBoundingClientRect();
    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;

    // Scale to grid coordinates
    const gridX = (canvasX / rect.width) * GRID_SIZE.width;
    // Note: Y coordinate should NOT be flipped since both canvas and WebGPU use top-left origin
    const gridY = (canvasY / rect.height) * GRID_SIZE.height;

    return { x: gridX, y: gridY };
  }, []);

  // Mouse event handlers
  const handleMouseDown = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isInitialized || !smokeSimulation.current) return;

      setIsMouseDown(true);
      const coords = canvasToGridCoords(event.clientX, event.clientY);
      if (coords) {
        // Add smoke on click
        smokeSimulation.current.addSmoke(coords.x, coords.y);
        previousMousePos.current = coords;
        lastInteractionTime.current = performance.now();
      }
    },
    [isInitialized, canvasToGridCoords]
  );

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isInitialized || !smokeSimulation.current || !isMouseDown) return;

      const coords = canvasToGridCoords(event.clientX, event.clientY);
      if (coords && previousMousePos.current) {
        const currentTime = performance.now();
        const deltaTime = (currentTime - lastInteractionTime.current) / 1000; // Convert to seconds

        // Add smoke continuously while dragging
        smokeSimulation.current.addSmoke(coords.x, coords.y);

        if (deltaTime > 0) {
          // Calculate velocity based on mouse movement with scaling
          const rawVelocityX =
            (coords.x - previousMousePos.current.x) / deltaTime;
          const rawVelocityY =
            (coords.y - previousMousePos.current.y) / deltaTime;

          // Scale down and clamp velocity to reasonable range (1-10)
          const velocityScale = 0.02; // Scale factor to reduce velocity
          const maxVelocity = 10;

          let velocityX = rawVelocityX * velocityScale;
          let velocityY = rawVelocityY * velocityScale;

          // Clamp velocity magnitude
          const velocityMagnitude = Math.sqrt(
            velocityX * velocityX + velocityY * velocityY
          );
          if (velocityMagnitude > maxVelocity) {
            const scale = maxVelocity / velocityMagnitude;
            velocityX *= scale;
            velocityY *= scale;
          }

          // Add velocity to the simulation
          smokeSimulation.current.addVelocity(
            coords.x,
            coords.y,
            velocityX,
            velocityY
          );

          previousMousePos.current = coords;
          lastInteractionTime.current = currentTime;
        }
      }
    },
    [isInitialized, isMouseDown, canvasToGridCoords]
  );

  const handleMouseUp = useCallback(() => {
    setIsMouseDown(false);
    previousMousePos.current = null;
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseDown(false);
    previousMousePos.current = null;
  }, []);

  // Visualization mode options
  const visualizationModes = [
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

  const handleModeChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const mode = parseInt(event.target.value) as ShaderMode;
      const modeOption = visualizationModes.find((m) => m.value === mode);
      if (modeOption) {
        setSelectedMode(mode);
        setSelectedTexture(modeOption.texture);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setSelectedMode, setSelectedTexture]
  );

  // Custom visualization mode dropdown
  const visualizationControl = (
    <div className="w-full">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Visualization Mode
      </label>
      <select
        value={selectedMode}
        onChange={handleModeChange}
        disabled={!isInitialized}
        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
      >
        {visualizationModes.map((mode) => (
          <option key={mode.value} value={mode.value}>
            {mode.label}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="p-5">
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
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
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
