import * as Tone from "tone";

export class ToneInitializationError extends Error {
  constructor(message: string) {
    super(`ToneJS initialization error: ${message}`);
  }
}

export class ToneUtils {
  private static initialized = false;

  /**
   * Initialize ToneJS audio context.
   *
   * WARNING: This method MUST ONLY be called during a user gesture event
   * (such as click, keydown, touchstart, etc.) due to browser autoplay policies.
   */
  public static async initialize(): Promise<void> {
    console.log("Initializing ToneJS");
    if (ToneUtils.initialized) return;

    try {
      await Tone.start();
      ToneUtils.initialized = true;
      console.log("ToneJS has been initialized");
    } catch (error) {
      throw new ToneInitializationError(`Failed to start ToneJS: ${error}`);
    }
  }

  /**
   * Check if ToneJS has been initialized
   */
  public static isInitialized(): boolean {
    return ToneUtils.initialized;
  }

  /**
   * Reset initialization state (useful for testing)
   */
  public static reset(): void {
    ToneUtils.initialized = false;
  }
}
