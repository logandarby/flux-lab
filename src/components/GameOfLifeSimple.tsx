import React, { useEffect, useRef, useState } from "react";
import { SmokeEngine } from "@/webgpu/game-of-life-engine";
import { DEFAULT_CONFIG } from "@/webgpu/types";
import { WebGPUError } from "@/webgpu/webgpu-utils";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Separator } from "./ui/separator";
import {
  PlayIcon,
  PauseIcon,
  SkipForwardIcon,
  RotateCcwIcon,
} from "lucide-react";

export const GameOfLifeSimple: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<SmokeEngine | null>(null);
  const [status, setStatus] = useState<string>("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  const config = {
    ...DEFAULT_CONFIG,
  };

  useEffect(() => {
    let mounted = true;

    const initEngine = async () => {
      if (!canvasRef.current) return;

      try {
        const engine = new SmokeEngine();
        await engine.initialize(canvasRef.current, config);

        if (mounted) {
          engineRef.current = engine;
          engine.start();
          setStatus("Running");
          setIsRunning(true);
          setError(null);
        } else {
          engine.destroy();
        }
      } catch (err) {
        if (mounted) {
          const errorMessage =
            err instanceof WebGPUError
              ? err.message
              : `Error: ${
                  err instanceof Error ? err.message : "Unknown error"
                }`;
          setError(errorMessage);
          setStatus("Error");
        }
      }
    };

    initEngine();

    return () => {
      mounted = false;
      if (engineRef.current) {
        engineRef.current.destroy();
        engineRef.current = null;
      }
    };
  }, []);

  const handleStart = () => {
    if (engineRef.current && !isRunning) {
      engineRef.current.start();
      setIsRunning(true);
      setStatus("Running");
    }
  };

  const handleStop = () => {
    if (engineRef.current && isRunning) {
      engineRef.current.stop();
      setIsRunning(false);
      setStatus("Stopped");
    }
  };

  const handleStep = () => {
    if (engineRef.current && !isRunning) {
      engineRef.current.step();
    }
  };

  const handleRestart = async () => {
    if (engineRef.current) {
      engineRef.current.destroy();
      engineRef.current = null;
    }

    setStatus("Restarting...");
    setIsRunning(false);
    setError(null);

    // Small delay to ensure cleanup
    setTimeout(async () => {
      if (!canvasRef.current) return;

      try {
        const engine = new SmokeEngine();
        await engine.initialize(canvasRef.current, config);

        engineRef.current = engine;
        engine.start();
        setStatus("Running");
        setIsRunning(true);
      } catch (err) {
        const errorMessage =
          err instanceof WebGPUError
            ? err.message
            : `Error: ${err instanceof Error ? err.message : "Unknown error"}`;
        setError(errorMessage);
        setStatus("Error");
      }
    }, 100);
  };

  return (
    <Card className="w-fit mx-auto">
      <CardHeader className="text-center">
        <CardTitle>Conway's Game of Life</CardTitle>
        <CardDescription>
          {config.gridSize}×{config.gridSize} grid • WebGPU Compute Shaders
        </CardDescription>
      </CardHeader>

      <CardContent>
        <div className="flex gap-6 items-start">
          {/* Canvas on the left */}
          <div className="flex-shrink-0">
            <canvas
              ref={canvasRef}
              width={512}
              height={512}
              className="border-2 border-border rounded-lg shadow-lg"
            />
          </div>

          {/* Controls on the right */}
          <div className="flex flex-col space-y-4 min-w-48">
            {/* Error display */}
            {error && (
              <div className="bg-destructive/10 border border-destructive/20 text-destructive rounded-md p-3 text-sm">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Control buttons */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground mb-3">
                Controls
              </h3>

              <Button
                onClick={isRunning ? handleStop : handleStart}
                disabled={!!error}
                variant={isRunning ? "destructive" : "default"}
                size="sm"
                className="w-full justify-start"
              >
                {isRunning ? (
                  <>
                    <PauseIcon className="w-4 h-4" />
                    Stop
                  </>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4" />
                    Start
                  </>
                )}
              </Button>

              <Button
                onClick={handleStep}
                disabled={!!error || isRunning}
                variant="outline"
                size="sm"
                className="w-full justify-start"
              >
                <SkipForwardIcon className="w-4 h-4" />
                Step
              </Button>

              <Button
                onClick={handleRestart}
                disabled={!!error}
                variant="secondary"
                size="sm"
                className="w-full justify-start"
              >
                <RotateCcwIcon className="w-4 h-4" />
                Restart
              </Button>
            </div>

            <Separator />

            {/* Status section */}
            <div className="space-y-2 text-sm">
              <h3 className="text-muted-foreground inline-block">Status:</h3>
              &nbsp;
              <span
                className={`font-medium ${
                  status === "Running"
                    ? "text-green-600"
                    : status === "Error"
                    ? "text-destructive"
                    : "text-muted-foreground"
                }`}
              >
                {status}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
