import { useRef, useEffect, useCallback } from "react";
import { AudioVisualizationMediator } from "../core/AudioVisualizationMediator";
import { DEFAULT_AUDIO_VISUALIZATION_CONFIG } from "../config";
import type SmokeSimulation from "@/lib/simulation/core/SmokeSimulation";
import type { AudioPreset, AudioVisualizationConfig } from "../types";

/**
 * React hook for audio visualization integration
 */
export function useAudioVisualization(
  config: AudioVisualizationConfig = DEFAULT_AUDIO_VISUALIZATION_CONFIG
) {
  const mediatorRef = useRef<AudioVisualizationMediator | null>(null);

  // Initialize mediator
  useEffect(() => {
    mediatorRef.current = new AudioVisualizationMediator(config);
    return () => {
      mediatorRef.current?.destroy();
      mediatorRef.current = null;
    };
  }, [config]);

  const initialize = useCallback(
    async (
      canvas: HTMLCanvasElement,
      simulation: SmokeSimulation,
      audioPreset?: AudioPreset
    ) => {
      if (!mediatorRef.current) return;
      await mediatorRef.current.initialize(canvas, simulation, audioPreset);
    },
    []
  );

  const handleMouseDown = useCallback(
    async (clientX: number, clientY: number) => {
      await mediatorRef.current?.onMouseDown(clientX, clientY);
    },
    []
  );

  const handleMouseMove = useCallback(
    async (clientX: number, clientY: number) => {
      await mediatorRef.current?.onMouseMove(clientX, clientY);
    },
    []
  );

  const handleMouseUp = useCallback(
    async (clientX: number, clientY: number) => {
      await mediatorRef.current?.onMouseUp(clientX, clientY);
    },
    []
  );

  const handleMouseLeave = useCallback(() => {
    mediatorRef.current?.onMouseLeave();
  }, []);

  const reset = useCallback(() => {
    mediatorRef.current?.reset();
  }, []);

  const isAudioInitialized = useCallback(() => {
    return mediatorRef.current?.isAudioInitialized() ?? false;
  }, []);

  const setAudioPreset = useCallback((preset: AudioPreset) => {
    mediatorRef.current?.setAudioPreset(preset);
  }, []);

  return {
    initialize,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleMouseLeave,
    reset,
    isAudioInitialized,
    setAudioPreset,
  };
}
