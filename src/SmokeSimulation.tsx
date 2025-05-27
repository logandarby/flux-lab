import { useCallback, useRef } from "react";
import { useEffect, useState } from "react";
import textureShader from "./textureShader.wgsl?raw";
import divergenceShaderTemplate from "./shaders/divergenceShader.wgsl?raw";
import advectionShaderTemplate from "./shaders/advectionShader.wgsl?raw";

// Constants

const CANVAS_WIDTH = 512;
const CANVAS_HEIGHT = 512;
const GRID_SIZE = 16; // 8x8 grid
const FLOAT_BYTES = 4;
const WORKGROUP_SIZE = 8;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
// const SIMULATION_TIME_STEP_MS = 1000 / 60;

enum WebGPUErrorCode {
  WEBGPU_NOT_SUPPORTED,
  NO_ADAPTER,
  NO_DEVICE,
  NO_CONTEXT,
  NO_CANVAS,
  NO_RESOURCES,
}

interface WebGPUResources {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}

// Utils

/**
 * WebGPU utility functions for initialization and setup
 */
export class WebGPUError extends Error {
  constructor(message: string, public readonly code?: WebGPUErrorCode) {
    super(message);
    this.name = "WebGPUError";
  }
}

/**
 * Initialize WebGPU
 * @param canvas - The canvas element to initialize WebGPU on
 * @returns {Promise<{ device: GPUDevice; context: GPUCanvasContext; canvasFormat: GPUTextureFormat }>}
 */
async function initializeWebGPU(
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

/**
 * Template literal function to inject variables into WGSL shader strings
 * @param template - The shader template string with ${variableName} placeholders
 * @param variables - Object containing variable name-value pairs
 * @returns The processed shader string with variables injected
 */
function injectShaderVariables(
  template: string,
  variables: Record<string, string | number>
): string {
  return template.replace(/\$\{(\w+)\}/g, (_, variableName) => {
    if (variableName in variables) {
      return String(variables[variableName]);
    }
    throw new Error(
      `Variable '${variableName}' not found in shader template variables`
    );
  });
}

interface TextureSwapBuffer {
  front: GPUTexture;
  back: GPUTexture;
}

interface TextureBuffer {
  front: GPUTexture;
}

class TextureManager<TextureID extends string | number> {
  private readonly textures = new Map<
    TextureID,
    TextureSwapBuffer | TextureBuffer
  >();

  constructor(private readonly device: GPUDevice) {}

  public createTexture(
    name: TextureID,
    descriptor: GPUTextureDescriptor
  ): void {
    if (this.textures.has(name)) {
      throw new Error(`Texture ${name} already exists inside TextureManager!`);
    }
    this.textures.set(name, {
      front: this.device.createTexture(descriptor),
    });
  }

  public createPingPongTexture(
    name: TextureID,
    descriptor: GPUTextureDescriptor
  ): void {
    if (this.textures.has(name)) {
      throw new Error(`Texture ${name} already exists inside TextureManager!`);
    }
    const texture1 = this.device.createTexture(descriptor);
    const texture2 = this.device.createTexture(descriptor);
    this.textures.set(name, {
      front: texture1,
      back: texture2,
    });
  }

  public swap(name: TextureID): void {
    const textures = this.textures.get(name);
    if (!textures) {
      throw new Error(`Texture '${name}' not found`);
    }
    if (!("back" in textures)) {
      throw new Error(`Cannot swap static single texture ${name}`);
    }
    const { front, back } = textures;
    this.textures.set(name, {
      front: back,
      back: front,
    });
  }

  public getCurrentTexture(name: TextureID): GPUTexture {
    const frontTexture = this.textures.get(name);
    if (!frontTexture) {
      throw new Error(`Texture ${name} does not exist`);
    }
    return frontTexture.front;
  }

  public getBackTexture(name: TextureID): GPUTexture {
    const backTexture = this.textures.get(name);
    if (!backTexture) {
      throw new Error(`Texture ${name} does not exist`);
    }
    if (!("back" in backTexture)) {
      throw new Error(
        `Texture ${name} is not a ping-pong texture-- it has no back buffer, only a front`
      );
    }
    return backTexture.back;
  }
}

interface ComputePassConfig {
  readonly name: string;
  readonly shader: GPUShaderModule;
  readonly entryPoint: string;
}

abstract class ComputePass<TextureID extends string | number> {
  protected readonly pipeline: GPUComputePipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(
    protected readonly config: ComputePassConfig,
    protected readonly device: GPUDevice
  ) {
    const { pipeline, bindGroupLayout } = this.setupPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }

  private setupPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const bindGroupLayout = this.createBindGroupLayout();
    const pipeline = this.device.createComputePipeline({
      label: `${this.config.name} Pipeline`,
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: this.config.shader,
        entryPoint: this.config.entryPoint,
      },
    });
    return { pipeline, bindGroupLayout };
  }

  protected abstract createBindGroupLayout(): GPUBindGroupLayout;
  protected abstract createBindGroup(
    textureManager: TextureManager<TextureID>
  ): GPUBindGroup;

  public execute(
    pass: GPUComputePassEncoder,
    textureManager: TextureManager<TextureID>,
    workgroupCount: number
  ): void {
    const bindGroup = this.createBindGroup(textureManager);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
  }
}

interface RenderPassConfig {
  name: string;
  vertex: GPUVertexState;
  fragment: GPUFragmentState;
}

interface RenderPassDrawConfig {
  vertexCount: number;
  instanceCount?: number;
}

abstract class RenderPass<TextureID extends string | number> {
  protected readonly pipeline: GPURenderPipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(
    protected readonly config: RenderPassConfig,
    protected readonly device: GPUDevice
  ) {
    this.bindGroupLayout = this.createBindGroupLayout();
    this.pipeline = this.device.createRenderPipeline({
      label: `${this.config.name} Pipeline`,
      layout: this.device.createPipelineLayout({
        label: `${this.config.name} Pipeline Layout`,
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: this.config.vertex,
      fragment: this.config.fragment,
    });
  }

  protected abstract createBindGroupLayout(): GPUBindGroupLayout;
  protected abstract createBindGroup(
    textureManager?: TextureManager<TextureID>
  ): GPUBindGroup;

  public execute(
    pass: GPURenderPassEncoder,
    drawConfig: RenderPassDrawConfig,
    textureManager?: TextureManager<TextureID>
  ): void {
    const bindGroup = this.createBindGroup(textureManager);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(drawConfig.vertexCount, drawConfig.instanceCount);
  }
}

type TextureID = "divergence" | "velocity";

class DivergencePass extends ComputePass<TextureID> {
  constructor(device: GPUDevice) {
    super(
      {
        name: "divergence",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Divergence Shader",
          code: injectShaderVariables(divergenceShaderTemplate, {
            WORKGROUP_SIZE,
          }),
        }),
      },
      device
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Divergence Compute Bind Group Layout",
      entries: [
        {
          // Velocity In
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
          },
        },
        {
          // Divergence Out
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "r32float",
            access: "write-only",
            viewDimension: "2d",
          },
        },
      ],
    });
  }

  protected createBindGroup(
    textureManager: TextureManager<TextureID>
  ): GPUBindGroup {
    return this.device.createBindGroup({
      label: `${this.config.name} BindGroup`,
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture("velocity").createView(),
        },
        {
          binding: 1,
          resource: textureManager.getCurrentTexture("divergence").createView(),
        },
      ],
    });
  }
}

class AdvectionPass extends ComputePass<TextureID> {

  private readonly linearSampler;

  constructor(device: GPUDevice) {
    super({
      name: "Advection Pass",
      entryPoint: "compute_main",
      shader: device.createShaderModule({
        label: "Advection Shader",
        code: injectShaderVariables(advectionShaderTemplate, {WORKGROUP_SIZE}),
      })
    }, device);
    this.linearSampler = device.createSampler({
      label: "Linear Sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Advection Bind Group Layout",
      entries: [
        // Velocity
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
          }
        },
        // Advection In
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "float",
            viewDimension: "2d",
          }
        },
        // Advection Out
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "rgba16float",
            access: "write-only",
            viewDimension: "2d",
          }
        },
        // Sampler
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          sampler: {
            type: "filtering"
          }
        }
      ]
    })
  }

  protected createBindGroup(textureManager: TextureManager<TextureID>): GPUBindGroup {
    return this.device.createBindGroup({
      label: "Advection Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture("velocity").createView()
        },
        {
          binding: 1,
          resource: textureManager.getCurrentTexture("velocity").createView()
        },
        {
          binding: 2,
          resource: textureManager.getBackTexture("velocity").createView()
        },
        {
          binding: 3,
          resource: this.linearSampler,
        }
      ]
    })
  }

}

class RenderTexturePass extends RenderPass<TextureID> {
  private readonly sampler: GPUSampler;

  constructor(config: RenderPassConfig, device: GPUDevice) {
    super(config, device);
    this.sampler = this.device.createSampler({
      label: "Texture Sampler"
    });
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Texture Rendering Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
            multisampled: false,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {
            type: "non-filtering",
          },
        },
      ]
    });
  }

  protected createBindGroup(
    textureManager: TextureManager<TextureID>
  ): GPUBindGroup {
    return this.device.createBindGroup({
      label: "Smoke Texture Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture("velocity").createView(),
        },
        {
          binding: 1,
          resource: this.sampler,
        },
      ],
    });
  }
}

function initializeVelocityField(
  velocityTexture: GPUTexture,
  device: GPUDevice
): void {
  const _ = [0.0, 0.0];
  const v = [1.0, 1.0];
  // prettier-ignore
  const velocityData = new Float32Array([
    _, _, _, _, _, _, v, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, v, v, v, _, _, _, _, _, _, _,
    _, v, _, _, _, _, _, v, v, v, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, v, v, v, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
  ].flat());
  device.queue.writeTexture(
    { texture: velocityTexture },
    velocityData,
    {
      bytesPerRow: GRID_SIZE * v.length * FLOAT_BYTES,
    },
    { width: GRID_SIZE, height: GRID_SIZE }
  );
}

class SmokeSimulation {
  private resources: WebGPUResources | null = null;
  private textureManager: TextureManager<TextureID> | null = null;
  private divergencePass: DivergencePass | null = null;
  private advectionPass: AdvectionPass | null = null;
  private renderingPass: RenderTexturePass | null = null;

  public async initialize(canvasRef: React.RefObject<HTMLCanvasElement>) {
    if (!canvasRef.current) {
      throw new WebGPUError(
        "Could not initialize WebGPU: Canvas not found",
        WebGPUErrorCode.NO_CANVAS
      );
    }
    this.resources = await initializeWebGPU(canvasRef.current);

    // Initialize all the simulation components
    this.textureManager = new TextureManager<TextureID>(this.resources.device);

    this.textureManager.createPingPongTexture("velocity", {
      label: "Velocity Texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "rgba16float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    initializeVelocityField(
      this.textureManager.getCurrentTexture("velocity"),
      this.resources.device
    );

    this.textureManager.createTexture("divergence", {
      label: "Divergence texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "r32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    this.divergencePass = new DivergencePass(this.resources.device);
    this.advectionPass = new AdvectionPass(this.resources.device);
    const textureShaderModule = this.resources.device.createShaderModule({
      label: "Texture Shader",
      code: textureShader,
    });
    this.renderingPass = new RenderTexturePass({
      name: "Texture Rendering",
        vertex: {
          module: textureShaderModule,
          entryPoint: "vertex_main",
        },
        fragment: {
          module: textureShaderModule,
          entryPoint: "fragment_main",
          targets: [
            {
              format: this.resources.canvasFormat,
            },
          ],
        },
    }, this.resources.device);

    // Render the initial state
    this.step({ renderOnly: true });
  }

  public step({ renderOnly }: { renderOnly: boolean } = { renderOnly: false }) {
    if (!this.resources || !this.textureManager || !this.advectionPass || !this.renderingPass) {
      throw new WebGPUError(
        "Could not step simulation: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    if (!renderOnly) {
      // Compute Pass (Advection)
      const advectionPassEncoder = commandEncoder.beginComputePass({
        label: "Advection Compute pass",
      });
      this.advectionPass.execute(
        advectionPassEncoder,
        this.textureManager,
        WORKGROUP_COUNT,
      );
      advectionPassEncoder.end();
      this.textureManager.swap("velocity");
    }

    // Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass({
      label: "Texture Render Pass",
      colorAttachments: [
        {
          view: this.resources.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        },
      ],
    });

    this.renderingPass.execute(renderPassEncoder, {
      vertexCount: 6,
    }, this.textureManager);

    renderPassEncoder.end();
    this.resources.device.queue.submit([commandEncoder.finish()]);
  }
}

function SmokeSimulationComponent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smokeSimulation = useRef<SmokeSimulation | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    const runSimulation = async () => {
      if (!canvasRef.current || !!smokeSimulation.current) {
        return;
      }
      console.log("Initializing Smoke Simulation");
      smokeSimulation.current = new SmokeSimulation();
      try {
        await smokeSimulation.current.initialize(canvasRef);
        setIsInitialized(true);
        setInitError(null);
      } catch (error) {
        console.error("Failed to initialize smoke simulation:", error);
        smokeSimulation.current = null;
        setIsInitialized(false);
        setInitError(error instanceof Error ? error.message : "Unknown error");
      }
    };
    runSimulation();
  }, []);

  const handleStep = useCallback(() => {
    if (smokeSimulation.current && isInitialized) {
      try {
        smokeSimulation.current.step();
      } catch (error) {
        console.error("Failed to step simulation:", error);
      }
    }
  }, [isInitialized, smokeSimulation.current]);

  return (
    <div className="p-5">
      <div className="max-w-2xl mx-auto flex flex-col items-center gap-4">
        <canvas width={CANVAS_WIDTH} height={CANVAS_HEIGHT} ref={canvasRef} />
        {initError && (
          <div className="text-red-500 text-sm max-w-md text-center">
            Error: {initError}
          </div>
        )}
        <button
          onClick={handleStep}
          className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-lg transition-colors duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed"
          disabled={!isInitialized}
        >
          Step
        </button>
        <div className="text-sm text-gray-600">
          Status: {isInitialized ? "Ready" : "Initializing..."}
        </div>
      </div>
    </div>
  );
}

export default SmokeSimulationComponent;
