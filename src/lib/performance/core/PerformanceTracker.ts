import { RollingAverage } from "./RollingAverage";

export interface PerformanceMetrics {
  js: number;
  fps: number;
  timestep: number;
  gpu: number;
  gpuSupported: boolean;
}

export class PerformanceTracker {
  private jsAverage = new RollingAverage();
  private fpsAverage = new RollingAverage();
  private timestepAverage = new RollingAverage();
  private gpuAverage = new RollingAverage();

  private timers = new Map<string, number>();
  private lastFrameTime = performance.now();
  private gpuSupported = false;

  constructor(gpuSupported: boolean = false) {
    this.gpuSupported = gpuSupported;
  }

  setGpuSupported(supported: boolean): void {
    this.gpuSupported = supported;
  }

  startTimer(name: string): void {
    this.timers.set(name, performance.now());
  }

  endTimer(name: string): number {
    const startTime = this.timers.get(name);
    if (!startTime) return 0;

    const elapsed = performance.now() - startTime;
    this.timers.delete(name);
    return elapsed;
  }

  recordJS(timeMs: number): void {
    this.jsAverage.addSample(timeMs);
  }

  recordGPU(timeMicroseconds: number): void {
    this.gpuAverage.addSample(timeMicroseconds);
  }

  recordTimestep(timestepSeconds: number): void {
    this.timestepAverage.addSample(timestepSeconds * 1000); // Convert to milliseconds for display
  }

  recordFrame(): void {
    const now = performance.now();
    const deltaTime = (now - this.lastFrameTime) / 1000;
    if (deltaTime > 0) {
      this.fpsAverage.addSample(1 / deltaTime);
    }
    this.lastFrameTime = now;
  }

  getMetrics(): PerformanceMetrics {
    return {
      js: this.jsAverage.get(),
      fps: this.fpsAverage.get(),
      timestep: this.timestepAverage.get(),
      gpu: this.gpuAverage.get(),
      gpuSupported: this.gpuSupported,
    };
  }

  clear(): void {
    this.jsAverage.clear();
    this.fpsAverage.clear();
    this.timestepAverage.clear();
    this.gpuAverage.clear();
    this.timers.clear();
  }
}
