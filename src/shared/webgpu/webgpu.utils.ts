export enum WebGPUErrorCode {
  WEBGPU_NOT_SUPPORTED,
  NO_ADAPTER,
  NO_DEVICE,
  NO_CONTEXT,
  NO_CANVAS,
  NO_RESOURCES,
}

export interface WebGPUResources {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}

/**
 * WebGPU utility functions for initialization and setup
 */
export class WebGPUError extends Error {
  constructor(
    message: string,
    public readonly code?: WebGPUErrorCode
  ) {
    super(message);
    this.name = "WebGPUError";
  }
}

/**
 * Initialize WebGPU
 * @param canvas - The canvas element to initialize WebGPU on
 * @returns {Promise<{ device: GPUDevice; context: GPUCanvasContext; canvasFormat: GPUTextureFormat }>}
 */
export async function initializeWebGPU(
  canvas: HTMLCanvasElement
): Promise<WebGPUResources> {
  // Check WebGPU support
  if (!navigator.gpu) {
    throw new WebGPUError(
      "WebGPU is not supported on this browser. Please use Google Chrome or Microsoft Edge to run this application",
      WebGPUErrorCode.WEBGPU_NOT_SUPPORTED
    );
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new WebGPUError("No GPU Adapter found", WebGPUErrorCode.NO_ADAPTER);
  }

  // Request device
  const device = await adapter.requestDevice();
  if (!device) {
    throw new WebGPUError("No GPU Device Found", WebGPUErrorCode.NO_DEVICE);
  }

  // Setup canvas context
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new WebGPUError(
      "Could not get Canvas Context",
      WebGPUErrorCode.NO_CONTEXT
    );
  }

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: canvasFormat,
  });

  return { device, context, canvasFormat };
}
