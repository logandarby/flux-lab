// Core
export { AudioVisualizationMediator } from "./core/AudioVisualizationMediator";

// Utils
export { ToneUtils } from "./utils/ToneUtils";

// Presets
export {
  PentatonicSynthPreset,
  type Note,
  type NoteVelocity,
} from "./presets/PentatonicSynthPreset";

// Hooks
export { useAudioVisualization } from "./hooks/useAudioVisualization";

// Configuration and Errors
export { DEFAULT_AUDIO_VISUALIZATION_CONFIG } from "./config";
export { AudioVisualizationError } from "./errors";

// Types
export type * from "./types";
