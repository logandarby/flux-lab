import { useCallback, useRef } from "react";
import { useEffect, useState } from "react";
import textureShader from "./shaders/textureShader.wgsl?raw";
import divergenceShaderTemplate from "./shaders/divergenceShader.wgsl?raw";
import advectionShaderTemplate from "./shaders/advectionShader.wgsl?raw";
import { ComputePass } from "./utils/ComputePass";
import {
  initializeWebGPU,
  injectShaderVariables,
  WebGPUError,
  WebGPUErrorCode,
  type WebGPUResources,
} from "./utils/webgpu.utils";
import { TextureManager } from "./utils/TextureManager";
import { RenderPass, type RenderPassConfig } from "./utils/RenderPass";
import SimulationControls from "./components/ui/SimulationControls";

// Constants

const CANVAS_WIDTH = 512;
const CANVAS_HEIGHT = 512;
const GRID_SIZE = 16; // 8x8 grid
// const FLOAT_BYTES = 4;
const WORKGROUP_SIZE = 8;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
// const SIMULATION_TIME_STEP_MS = 1000 / 60;

// Utils

export type SmokeTextureID = "divergence" | "velocity";

class DivergencePass extends ComputePass<SmokeTextureID> {
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
    textureManager: TextureManager<SmokeTextureID>
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

export class AdvectionPass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice) {
    super(
      {
        name: "Advection Pass",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Advection Shader",
          code: injectShaderVariables(advectionShaderTemplate, {
            WORKGROUP_SIZE,
          }),
        }),
      },
      device
    );
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
          },
        },
        // Advection In
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
          },
        },
        // Advection Out
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "rg32float",
            access: "write-only",
            viewDimension: "2d",
          },
        },
      ],
    });
  }

  protected createBindGroup(
    textureManager: TextureManager<SmokeTextureID>
  ): GPUBindGroup {
    return this.device.createBindGroup({
      label: "Advection Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture("velocity").createView(),
        },
        {
          binding: 1,
          resource: textureManager.getCurrentTexture("velocity").createView(),
        },
        {
          binding: 2,
          resource: textureManager.getBackTexture("velocity").createView(),
        },
      ],
    });
  }
}

export class RenderTexturePass extends RenderPass<SmokeTextureID> {
  private readonly sampler: GPUSampler;

  constructor(config: RenderPassConfig, device: GPUDevice) {
    super(config, device);
    this.sampler = this.device.createSampler({
      label: "Texture Sampler",
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
      ],
    });
  }

  protected createBindGroup(
    textureManager: TextureManager<SmokeTextureID>
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
  const _ = [0.0, 0.0]; // RG channels for 32-bit float
  const v = [-1.0, 0.0]; // RG channels for 32-bit float
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
      bytesPerRow: GRID_SIZE * 2 * 4, // 2 channels × 4 bytes per 32-bit float
    },
    { width: GRID_SIZE, height: GRID_SIZE }
  );
}

class SmokeSimulation {
  private resources: WebGPUResources | null = null;
  private textureManager: TextureManager<SmokeTextureID> | null = null;
  // @ts-expect-error TODO: Will be implemented eventually
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
    this.textureManager = new TextureManager<SmokeTextureID>(
      this.resources.device
    );

    this.textureManager.createPingPongTexture("velocity", {
      label: "Velocity Texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "rg32float",
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
    this.renderingPass = new RenderTexturePass(
      {
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
      },
      this.resources.device
    );

    // Render the initial state
    this.step({ renderOnly: true });
  }

  public step({ renderOnly }: { renderOnly: boolean } = { renderOnly: false }) {
    if (
      !this.resources ||
      !this.textureManager ||
      !this.advectionPass ||
      !this.renderingPass
    ) {
      throw new WebGPUError(
        "Could not step simulation: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    // if (true) {
    // TODO: Put back
    if (!renderOnly) {
      // Compute Pass (Advection)
      const advectionPassEncoder = commandEncoder.beginComputePass({
        label: "Advection Compute pass",
      });
      this.advectionPass.execute(
        advectionPassEncoder,
        this.textureManager,
        WORKGROUP_COUNT
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

    this.renderingPass.execute(
      renderPassEncoder,
      {
        vertexCount: 6,
      },
      this.textureManager
    );

    renderPassEncoder.end();
    this.resources.device.queue.submit([commandEncoder.finish()]);
  }
}

function SmokeSimulationComponent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smokeSimulation = useRef<SmokeSimulation | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
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

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !isInitialized || !smokeSimulation.current) {
      return;
    }

    const animate = () => {
      if (smokeSimulation.current && isPlaying) {
        try {
          smokeSimulation.current.step();
          animationFrameRef.current = requestAnimationFrame(animate);
        } catch (error) {
          console.error("Failed to step simulation:", error);
          setIsPlaying(false);
        }
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isPlaying, isInitialized]);

  const handleStep = useCallback(() => {
    if (smokeSimulation.current && isInitialized) {
      try {
        smokeSimulation.current.step();
      } catch (error) {
        console.error("Failed to step simulation:", error);
      }
    }
  }, [isInitialized]);

  const handleRestart = useCallback(async () => {
    if (!smokeSimulation.current || !isInitialized) {
      return;
    }

    try {
      // Reinitialize the velocity field
      if (smokeSimulation.current && canvasRef.current) {
        await smokeSimulation.current.initialize(canvasRef);
      }
    } catch (error) {
      console.error("Failed to restart simulation:", error);
    }
  }, [isInitialized]);

  return (
    <div className="p-5">
      <div className="max-w-6xl mx-auto flex flex-col items-center gap-6">
        {/* Title and Description */}
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            Smoke Simulation
          </h2>
          <p className="text-gray-600 text-sm max-w-md">
            WebGPU-based fluid simulation with advection and divergence
            computation
          </p>
        </div>

        {initError && (
          <div className="text-red-500 text-sm max-w-md text-center bg-red-50 p-3 rounded-lg border border-red-200">
            <strong>Error:</strong> {initError}
          </div>
        )}

        {/* Main Content Area - Canvas and Controls */}
        <div className="flex flex-col md:flex-row items-center md:items-start gap-6 w-full justify-center">
          {/* Canvas */}
          <div className="flex-shrink-0">
            <canvas
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              ref={canvasRef}
              className="border border-gray-300 rounded-lg shadow-lg"
            />
          </div>

          {/* Controls */}
          <SimulationControls
            isInitialized={isInitialized}
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            onStep={handleStep}
            onRestart={handleRestart}
            title="Smoke Simulation"
          />
        </div>
      </div>
    </div>
  );
}

export default SmokeSimulationComponent;
