import type SmokeSimulation from "@/lib/simulation/core/SmokeSimulation";
import type { SmokeTextureExports } from "@/lib/simulation/core/SmokeSimulation";
import type {
  AudioPreset,
  CoordinateData,
  AudioVisualizationConfig,
} from "../types";
import { AudioVisualizationError } from "../errors";
import { PentatonicSynthPreset } from "../presets/PentatonicSynthPreset";
import { DEFAULT_AUDIO_VISUALIZATION_CONFIG } from "../config";
import { ToneUtils } from "../utils/ToneUtils";

/**
 * Mediator class to handle audio-visual interactions between smoke simulation and audio engine
 */
export class AudioVisualizationMediator {
  private simulation: SmokeSimulation | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private canvasBounds: DOMRect | null = null;
  private canvasBoundsUpdateTime = 0;
  private audioInitialized = false;

  private sampledTextureData: SmokeTextureExports | null = null;
  private samplingInterval: number | null = null;
  private isMouseDown = false;

  private audioPreset: AudioPreset | null = null;

  private readonly config: Required<AudioVisualizationConfig>;

  constructor(config: Partial<AudioVisualizationConfig> = {}) {
    this.config = {
      ...DEFAULT_AUDIO_VISUALIZATION_CONFIG,
      ...config,
    };
  }

  /**
   * Initialize the mediator with canvas, simulation, and audio preset
   */
  async initialize(
    canvas: HTMLCanvasElement,
    simulation: SmokeSimulation,
    audioPreset?: AudioPreset
  ): Promise<void> {
    console.log("Initializing Audio Visualization Mediator");
    this.canvas = canvas;
    this.simulation = simulation;

    // Set up audio preset (default to pentatonic if none provided)
    this.audioPreset = audioPreset ?? new PentatonicSynthPreset();

    // Set up user gesture listeners for audio initialization
    this.setupAudioInitialization();
  }

  /**
   * Start texture sampling at the configured interval
   */
  startSampling(): void {
    if (this.samplingInterval || !this.simulation) return;

    this.sampleTextures(); // Sample immediately
    this.samplingInterval = window.setInterval(
      () => this.sampleTextures(),
      this.config.samplingIntervalMs
    );
  }

  /**
   * Stop texture sampling
   */
  stopSampling(): void {
    if (this.samplingInterval) {
      clearInterval(this.samplingInterval);
      this.samplingInterval = null;
    }
  }

  /**
   * Handle mouse down events
   */
  async onMouseDown(clientX: number, clientY: number): Promise<void> {
    if (!this.audioPreset) return;

    this.isMouseDown = true;
    const coords = this.getCoordinates(clientX, clientY);
    if (!coords) return;

    this.startSampling();
    await this.audioPreset.onMouseDown(
      coords.normalizedX,
      coords.normalizedY,
      this.sampledTextureData
    );
  }

  /**
   * Handle mouse move events
   */
  async onMouseMove(clientX: number, clientY: number): Promise<void> {
    if (!this.audioInitialized || !this.audioPreset || !this.isMouseDown)
      return;

    const coords = this.getCoordinates(clientX, clientY);
    if (!coords) return;

    await this.audioPreset.onMouseMove(
      coords.normalizedX,
      coords.normalizedY,
      this.sampledTextureData
    );
  }

  /**
   * Handle mouse up events
   */
  async onMouseUp(clientX: number, clientY: number): Promise<void> {
    if (!this.audioInitialized || !this.audioPreset) return;

    this.isMouseDown = false;
    const coords = this.getCoordinates(clientX, clientY);
    if (!coords) return;

    await this.audioPreset.onMouseUp(
      coords.normalizedX,
      coords.normalizedY,
      this.sampledTextureData
    );
  }

  /**
   * Handle mouse leave events
   */
  onMouseLeave(): void {
    this.isMouseDown = false;
    this.stopSampling();
    this.audioPreset?.reset();
  }

  /**
   * Reset the audio visualization state
   */
  reset(): void {
    this.stopSampling();
    this.isMouseDown = false;
    this.audioPreset?.reset();
    this.sampledTextureData = null;
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    this.stopSampling();
    this.removeAudioInitializationListeners();
    this.simulation = null;
    this.canvas = null;
    this.audioPreset = null;
    this.sampledTextureData = null;
    this.audioInitialized = false;
  }

  /**
   * Check if audio is initialized
   */
  isAudioInitialized(): boolean {
    return this.audioInitialized;
  }

  /**
   * Set a new audio preset
   */
  setAudioPreset(preset: AudioPreset): void {
    this.audioPreset?.reset();
    this.audioPreset = preset;
  }

  private async sampleTextures(): Promise<void> {
    if (!this.simulation) return;

    try {
      this.sampledTextureData = await this.simulation.sampleTextures(
        this.config.textureSampleDownscale
      );
    } catch (error) {
      console.warn("Failed to sample textures:", error);
    }
  }

  private getCoordinates(
    clientX: number,
    clientY: number
  ): CoordinateData | null {
    const bounds = this.getCachedCanvasBounds();
    if (!bounds) return null;

    const canvasX = clientX - bounds.left;
    const canvasY = clientY - bounds.top;

    const normalizedX = canvasX / bounds.width;
    // Flip Y coordinate to match texture coordinate system (top = 1, bottom = 0)
    const normalizedY = 1.0 - canvasY / bounds.height;

    // Validate coordinates
    if (
      normalizedX < 0 ||
      normalizedX > 1 ||
      normalizedY < 0 ||
      normalizedY > 1
    ) {
      return null;
    }

    // Convert to grid coordinates if needed
    const gridSize = this.simulation?.getGridSize();
    const gridX = normalizedX * (gridSize?.width ?? 1);
    const gridY = normalizedY * (gridSize?.height ?? 1);

    return {
      normalizedX,
      normalizedY,
      gridX,
      gridY,
    };
  }

  private getCachedCanvasBounds(): DOMRect | null {
    const now = performance.now();
    if (!this.canvasBounds || now - this.canvasBoundsUpdateTime > 100) {
      if (this.canvas) {
        this.canvasBounds = this.canvas.getBoundingClientRect();
        this.canvasBoundsUpdateTime = now;
      }
    }
    return this.canvasBounds;
  }

  private setupAudioInitialization(): void {
    if (!this.canvas) {
      throw new AudioVisualizationError("Canvas not initialized");
    }

    const handleUserGesture = async () => {
      if (this.audioInitialized) return;

      try {
        await ToneUtils.initialize();
        this.audioInitialized = true;
        console.log("Audio engine initialized successfully");
        this.removeAudioInitializationListeners();
      } catch (error) {
        console.warn("Failed to initialize audio engine:", error);
      }
    };

    // Store bound handlers for cleanup
    this.boundHandlers = {
      click: handleUserGesture,
      mousedown: handleUserGesture,
      keydown: handleUserGesture,
    };

    const canvas = this.canvas;
    canvas.addEventListener("click", this.boundHandlers.click, { once: true });
    canvas.addEventListener("mousedown", this.boundHandlers.mousedown, {
      once: true,
    });
    canvas.addEventListener("keydown", this.boundHandlers.keydown, {
      once: true,
    });
    document.addEventListener("keydown", this.boundHandlers.keydown, {
      once: true,
    });
  }

  private boundHandlers: Record<string, () => void> = {};

  private removeAudioInitializationListeners(): void {
    if (!this.canvas) return;

    const canvas = this.canvas;
    canvas.removeEventListener("click", this.boundHandlers.click);
    canvas.removeEventListener("mousedown", this.boundHandlers.mousedown);
    canvas.removeEventListener("keydown", this.boundHandlers.keydown);
    document.removeEventListener("keydown", this.boundHandlers.keydown);
  }
}
