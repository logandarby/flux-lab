import { RollingAverage } from "./RollingAverage";

export interface PerformanceMetrics {
  js: number;
  fps: number;
}

export class PerformanceTracker {
  private jsAverage = new RollingAverage();
  private fpsAverage = new RollingAverage();

  private timers = new Map<string, number>();
  private lastFrameTime = performance.now();

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
    };
  }

  clear(): void {
    this.jsAverage.clear();
    this.fpsAverage.clear();
    this.timers.clear();
  }
}
