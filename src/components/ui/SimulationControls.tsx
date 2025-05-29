import { useCallback, useEffect } from "react";
import { Button } from "./button";
import { cn } from "@/lib/utils";

interface SimulationControlsProps {
  isInitialized: boolean;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
  onStep?: () => void;
  onRestart?: () => Promise<void>;
  title: string;
  customControls?: React.ReactNode;
}

function SimulationControls({
  isInitialized,
  isPlaying,
  setIsPlaying,
  onStep,
  onRestart,
  title,
  customControls,
}: SimulationControlsProps) {
  // Get status text and styling
  const getStatus = () => {
    if (!isInitialized)
      return { text: "Initializing...", type: "loading" as const };
    if (isPlaying) return { text: "Playing", type: "playing" as const };
    return { text: "Ready", type: "ready" as const };
  };

  const getStatusColor = (type: "loading" | "playing" | "ready") => {
    switch (type) {
      case "playing":
        return "bg-green-100 text-green-800 border-green-200";
      case "ready":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "loading":
        return "bg-gray-100 text-gray-800 border-gray-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const handleStep = useCallback(() => {
    if (isInitialized && onStep) {
      try {
        onStep();
      } catch (error) {
        console.error(`Failed to step ${title}:`, error);
      }
    }
  }, [isInitialized, onStep, title]);

  const handlePlay = useCallback(() => {
    if (isInitialized && onStep) {
      setIsPlaying(true);
    }
  }, [isInitialized, onStep, setIsPlaying]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
  }, [setIsPlaying]);

  const handleTogglePlayPause = useCallback(() => {
    if (isPlaying) {
      handlePause();
    } else {
      handlePlay();
    }
  }, [isPlaying, handlePlay, handlePause]);

  const handleRestart = useCallback(async () => {
    if (!isInitialized || !onRestart) {
      return;
    }

    setIsPlaying(false);

    try {
      await onRestart();
    } catch (error) {
      console.error(`Failed to restart ${title}:`, error);
    }
  }, [isInitialized, onRestart, title, setIsPlaying]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't trigger if user is typing in an input field
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.target instanceof HTMLSelectElement
      ) {
        return;
      }

      switch (event.code) {
        case "Space":
          event.preventDefault();
          if (isInitialized && onStep) {
            handleTogglePlayPause();
          }
          break;
        case "KeyR":
          // Only handle restart if no modifier keys are pressed (allow Ctrl+R for refresh)
          if (
            !event.ctrlKey &&
            !event.metaKey &&
            !event.altKey &&
            !event.shiftKey
          ) {
            event.preventDefault();
            if (isInitialized && onRestart) {
              handleRestart();
            }
          }
          break;
        case "KeyS":
          event.preventDefault();
          if (isInitialized && onStep && !isPlaying) {
            handleStep();
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    isInitialized,
    isPlaying,
    onStep,
    onRestart,
    handleTogglePlayPause,
    handleRestart,
    handleStep,
  ]);

  // Only show controls section if at least one control is available
  const hasControls = onStep || onRestart;
  const status = getStatus();

  return (
    <div className="flex flex-col items-center gap-4 md:min-w-[200px]">
      {/* Custom Controls */}
      {customControls && (
        <div className="w-full flex flex-col items-center gap-2">
          <h3 className="text-sm font-semibold text-gray-700 text-center mb-1">
            Settings
          </h3>
          {customControls}
        </div>
      )}

      {/* Control Buttons */}
      {hasControls && (
        <div className="flex flex-col gap-3 w-full">
          <h3 className="text-sm font-semibold text-gray-700 text-center mb-1">
            Controls
          </h3>
          {onStep && (
            <>
              <Button
                onClick={handlePlay}
                disabled={!isInitialized || isPlaying}
                variant="default"
                size="lg"
                title="Play simulation (Spacebar)"
                className="w-full"
              >
                Play
              </Button>
              <Button
                onClick={handlePause}
                disabled={!isInitialized || !isPlaying}
                variant="default"
                size="lg"
                title="Pause simulation (Spacebar)"
                className="w-full"
              >
                Pause
              </Button>
              <Button
                onClick={handleStep}
                disabled={!isInitialized || isPlaying}
                variant="default"
                size="lg"
                title="Step simulation once (S key)"
                className="w-full"
              >
                Step
              </Button>
            </>
          )}
          {onRestart && (
            <Button
              onClick={handleRestart}
              disabled={!isInitialized}
              variant="destructive"
              size="lg"
              title="Restart simulation (R key)"
              className="w-full"
            >
              Restart
            </Button>
          )}
        </div>
      )}

      {/* Status Badge */}
      <div className="flex flex-row items-center gap-2">
        <h3 className="text-sm font-semibold text-gray-700 inline-block">
          Status
        </h3>
        <span
          className={cn(
            "px-3 py-1 rounded-full text-xs font-medium border",
            getStatusColor(status.type)
          )}
        >
          {status.text}
        </span>
      </div>
    </div>
  );
}

export default SimulationControls;
