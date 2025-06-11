import type { AudioVisualizationConfig } from "../types";

/**
 * Default configuration for audio visualization
 */
export const DEFAULT_AUDIO_VISUALIZATION_CONFIG: Required<AudioVisualizationConfig> =
  {
    samplingIntervalMs: 100,
    textureSampleDownscale: 4,
  };
