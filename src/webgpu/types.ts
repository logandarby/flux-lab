export interface GOLConfig {
  gridSize: number;
  updateIntervalMs: number;
  workgroupSize: number;
}

export interface WebGPUResources {
  device: GPUDevice;
  context: GPUCanvasContext;
  vertexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  cellStateStorageBuffers: GPUBuffer[];
  bindGroups: GPUBindGroup[];
  cellPipeline: GPURenderPipeline;
  simulationPipeline: GPUComputePipeline;
}

export interface GOLEngine {
  initialize(
    canvas: HTMLCanvasElement,
    config: GOLConfig
  ): Promise<void>;
  start(): void;
  stop(): void;
  step(): void;
  isRunning(): boolean;
  destroy(): void;
}

export const DEFAULT_CONFIG: GOLConfig = {
  gridSize: 32,
  updateIntervalMs: 100,
  workgroupSize: 8,
};
