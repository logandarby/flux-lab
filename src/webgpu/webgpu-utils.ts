/**
 * WebGPU utility functions for initialization and setup
 */

export class WebGPUError extends Error {
  constructor(message: string, public readonly code?: string) {
    super(message);
    this.name = 'WebGPUError';
  }
}

export async function initializeWebGPU(canvas: HTMLCanvasElement): Promise<{
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}> {
  // Check WebGPU support
  if (!navigator.gpu) {
    throw new WebGPUError(
      "WebGPU is not supported on this browser. Please use Google Chrome or Microsoft Edge to run this application",
      "WEBGPU_NOT_SUPPORTED"
    );
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new WebGPUError("No GPU Adapter found", "NO_ADAPTER");
  }

  // Request device
  const device = await adapter.requestDevice();
  if (!device) {
    throw new WebGPUError("No GPU Device Found", "NO_DEVICE");
  }

  // Setup canvas context
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new WebGPUError("Could not get Canvas Context", "NO_CONTEXT");
  }

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: canvasFormat,
  });

  return { device, context, canvasFormat };
}

export function createVertexBuffer(device: GPUDevice, vertices: Float32Array): GPUBuffer {
  const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertices);
  return vertexBuffer;
}

export function createUniformBuffer(device: GPUDevice, gridSize: number): GPUBuffer {
  const uniformArray = new Float32Array([gridSize, gridSize]);
  const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
  return uniformBuffer;
}

export function createStorageBuffers(device: GPUDevice, gridSize: number): GPUBuffer[] {
  const cellStateArray = new Uint32Array(gridSize * gridSize);
  
  // Initialize with random state
  for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
  }

  const buffers = [
    device.createBuffer({
      label: "Cell State Storage Buffer A",
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      label: "Cell State Storage Buffer B",
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];

  device.queue.writeBuffer(buffers[0], 0, cellStateArray);
  return buffers;
}

export function createBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [
      // Grid uniform buffer
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
        buffer: {}
      },
      // Cell State Input Buffer
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      // Cell State Output Buffer
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      }
    ]
  });
}

export function createBindGroups(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  storageBuffers: GPUBuffer[]
): GPUBindGroup[] {
  return [
    device.createBindGroup({
      label: "Cell Bind Group A",
      layout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: storageBuffers[0] } },
        { binding: 2, resource: { buffer: storageBuffers[1] } }
      ]
    }),
    device.createBindGroup({
      label: "Cell Bind Group B",
      layout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: storageBuffers[1] } },
        { binding: 2, resource: { buffer: storageBuffers[0] } }
      ]
    })
  ];
} 