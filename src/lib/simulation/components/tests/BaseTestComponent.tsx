import { useCallback, useRef, useEffect, useState } from "react";
import SimulationControls from "../SimulationControls";

const CANVAS_DEFAULT = 512;

interface BaseTestProps {
  canvasHeight?: number;
  canvasWidth?: number;
  title: string;
  onInitialize?: (
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => Promise<void>;
  onStep?: () => void;
  onRestart?: (canvasRef: React.RefObject<HTMLCanvasElement>) => Promise<void>;
  customControls?: (
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => React.ReactNode;
}

function BaseTestComponent({
  title,
  onInitialize,
  onStep,
  onRestart,
  canvasHeight = CANVAS_DEFAULT,
  canvasWidth = CANVAS_DEFAULT,
  customControls,
}: BaseTestProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);

  // Initialize simulation
  useEffect(() => {
    const runSimulation = async () => {
      if (!canvasRef.current || isInitialized) {
        return;
      }
      console.log(`Initializing ${title}`);
      try {
        if (onInitialize) {
          await onInitialize(canvasRef);
        }
        setIsInitialized(true);
        setInitError(null);
      } catch (error) {
        console.error(`Failed to initialize ${title}:`, error);
        setIsInitialized(false);
        setInitError(error instanceof Error ? error.message : "Unknown error");
      }
    };
    runSimulation();
  }, [title, onInitialize, isInitialized]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !isInitialized || !onStep) {
      return;
    }

    const animate = () => {
      if (isPlaying && isInitialized && onStep) {
        try {
          onStep();
          animationFrameRef.current = requestAnimationFrame(animate);
        } catch (error) {
          console.error(`Failed to step ${title}:`, error);
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
  }, [isPlaying, isInitialized, onStep, title]);

  const handleStep = useCallback(() => {
    if (isInitialized && onStep) {
      try {
        onStep();
      } catch (error) {
        console.error(`Failed to step ${title}:`, error);
      }
    }
  }, [isInitialized, onStep, title]);

  const handleRestart = useCallback(async () => {
    if (!isInitialized || !onRestart) {
      return;
    }

    try {
      await onRestart(canvasRef);
    } catch (error) {
      console.error(`Failed to restart ${title}:`, error);
    }
  }, [isInitialized, onRestart, title]);

  return (
    <div className="p-5">
      <div className="max-w-6xl mx-auto flex flex-col items-center gap-6">
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
              width={canvasWidth}
              height={canvasHeight}
              ref={canvasRef}
              className="border border-gray-300 rounded-lg shadow-lg"
            />
          </div>

          {/* Controls */}
          <SimulationControls
            isInitialized={isInitialized}
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            onStep={handleStep}
            onRestart={handleRestart}
            title={title}
            customControls={customControls && customControls(canvasRef)}
          />
        </div>
      </div>
    </div>
  );
}

export default BaseTestComponent;
