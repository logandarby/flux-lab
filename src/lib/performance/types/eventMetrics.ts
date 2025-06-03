export type MouseEventData = {
  clientX: number;
  clientY: number;
  type: "mousedown" | "mousemove" | "mouseup";
  timestamp: number;
};

export interface EventPerformanceMetrics {
  mouseMoveCallsPerSecond?: number;
  mouseDownCallsPerSecond?: number;
  mouseUpCallsPerSecond?: number;
  keydownCallsPerSecond?: number;
  touchCallsPerSecond?: number;
}

export interface ExtendedPerformanceMetrics {
  system: {
    js: number;
    fps: number;
    timestep: number;
    gpu: number;
    gpuSupported: boolean;
  };
  events: EventPerformanceMetrics;
}
