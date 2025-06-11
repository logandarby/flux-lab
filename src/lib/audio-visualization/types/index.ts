import type { SmokeTextureExports } from "@/lib/simulation/core/SmokeSimulation";

/**
 * Audio preset interface for handling user interactions and generating audio
 */
export interface AudioPreset {
  onMouseDown(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void>;
  onMouseMove(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void>;
  onMouseUp(
    normalizedX: number,
    normalizedY: number,
    textureData: SmokeTextureExports | null
  ): Promise<void>;
  reset(): void;
}

/**
 * Coordinate data with both normalized and grid positions
 */
export interface CoordinateData {
  normalizedX: number;
  normalizedY: number;
  gridX: number;
  gridY: number;
}

/**
 * Configuration options for the mediator
 */
export interface AudioVisualizationConfig {
  samplingIntervalMs?: number;
  textureSampleDownscale?: number;
}
