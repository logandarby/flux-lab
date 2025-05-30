import { useCallback, useRef } from "react";
import { useEffect, useState } from "react";
import textureShader from "./shaders/textureShader.wgsl?raw";
import {
  initializeWebGPU,
  WebGPUError,
  WebGPUErrorCode,
  type WebGPUResources,
} from "./utils/webgpu.utils";
import { TextureManager } from "./utils/TextureManager";
import { RenderPass, type RenderPassConfig } from "./utils/RenderPass";
import SimulationControls from "./components/ui/SimulationControls";
import {
  AdvectionPass,
  DiffusionPass,
  DivergencePass,
  GradientSubtractionPass,
  PressurePass,
  UniformBufferUtils,
} from "./passes/SimulationPasses";

// Constants

const CANVAS_WIDTH = 2 ** 9;
const CANVAS_HEIGHT = CANVAS_WIDTH;
const GRID_SIZE = 16 * 7; // 16x16 grid
const GRID_SCALE = 1;
const WORKGROUP_SIZE = 8;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
// const VISCOSITY = 1;
const DIFFUSION_FACTOR = 0.02; // Smaller = slower diffusion
const VELOCITY_ADVECTION = 5; // Smaller = slower velocity advection
const TIMESTEP = 1.0 / 30.0;
const INIT_VELOCITY = [10, 10]; // TEMP
const DIFFUSION_ITERATIONS = 20;
const PRESSURE_ITERATIONS = 100;

// Utils

export type SmokeTextureID = "divergence" | "velocity" | "pressure";

export class RenderTexturePass extends RenderPass<SmokeTextureID> {
  constructor(config: RenderPassConfig<SmokeTextureID>, device: GPUDevice) {
    super(config, device);
  }
}

function initializeTextures(
  textureManager: TextureManager<SmokeTextureID>,
  device: GPUDevice
): void {
  const velocityTexture = textureManager.getCurrentTexture("velocity");
  // prettier-ignore
  const velocityData = Array(GRID_SIZE * GRID_SIZE).fill([0.0, 0.0]);
  const x = GRID_SIZE / 2;
  const y = GRID_SIZE / 2;
  const size = 8;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      velocityData[x + i + GRID_SIZE * (y + j)] = INIT_VELOCITY;
    }
  }
  device.queue.writeTexture(
    { texture: velocityTexture },
    new Float32Array(velocityData.flat()),
    {
      bytesPerRow: GRID_SIZE * 2 * 4, // 2 channels × 4 bytes per 32-bit float
    },
    { width: GRID_SIZE, height: GRID_SIZE }
  );
}

class SmokeSimulation {
  private resources: WebGPUResources | null = null;
  private textureManager: TextureManager<SmokeTextureID> | null = null;
  private advectionPass: AdvectionPass | null = null;
  private diffusionPass: DiffusionPass | null = null;
  private divergencePass: DivergencePass | null = null;
  private pressurePass: PressurePass | null = null;
  private gradientSubtractionPass: GradientSubtractionPass | null = null;
  private renderingPass: RenderTexturePass | null = null;
  private isInitialized = false;

  public async initialize(canvasRef: React.RefObject<HTMLCanvasElement>) {
    if (this.isInitialized) {
      console.warn("Simulation is already initialized");
      return;
    }
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

    // Create velocity texture (ping-pong)
    this.textureManager.createPingPongTexture("velocity", {
      label: "Velocity Texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    initializeTextures(this.textureManager, this.resources.device);

    // Create divergence texture
    this.textureManager.createTexture("divergence", {
      label: "Divergence texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "r32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create pressure texture (ping-pong)
    this.textureManager.createPingPongTexture("pressure", {
      label: "Pressure Texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Initialize simulation passes
    this.advectionPass = new AdvectionPass(
      this.resources.device,
      WORKGROUP_SIZE
    );
    this.diffusionPass = new DiffusionPass(
      this.resources.device,
      WORKGROUP_SIZE
    );
    this.divergencePass = new DivergencePass(
      this.resources.device,
      WORKGROUP_SIZE
    );
    this.pressurePass = new PressurePass(this.resources.device, WORKGROUP_SIZE);
    this.gradientSubtractionPass = new GradientSubtractionPass(
      this.resources.device,
      WORKGROUP_SIZE
    );

    // Create rendering pass
    const textureShaderModule = this.resources.device.createShaderModule({
      label: "Texture Shader",
      code: textureShader,
    });
    this.renderingPass = new RenderTexturePass(
      {
        outputTextureName: "velocity",
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

    this.isInitialized = true;

    // Render the initial state
    this.step({ renderOnly: true });
  }

  // @ts-expect-error TODO: Get rid og this
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public step({ renderOnly }: { renderOnly: boolean } = { renderOnly: false }) {
    if (
      !this.resources ||
      !this.textureManager ||
      !this.advectionPass ||
      !this.diffusionPass ||
      !this.divergencePass ||
      !this.pressurePass ||
      !this.gradientSubtractionPass ||
      !this.renderingPass ||
      !this.isInitialized
    ) {
      throw new WebGPUError(
        "Could not step simulation: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    // if (!renderOnly) {
    // TODO: Constant if is temporary
    // eslint-disable-next-line no-constant-condition
    if (true) {
      // Step 1: Advection
      const advectionUniforms = UniformBufferUtils.createAdvectionUniforms(
        TIMESTEP,
        VELOCITY_ADVECTION
      );
      const advectionBuffer = this.createUniformBuffer(
        advectionUniforms,
        "Advection"
      );

      const advectionPassEncoder = commandEncoder.beginComputePass({
        label: "Advection Compute Pass",
      });
      this.advectionPass.execute(
        advectionPassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: advectionBuffer,
        },
        WORKGROUP_COUNT
      );
      advectionPassEncoder.end();
      this.textureManager.swap("velocity");

      // Step 2: Diffusion
      const diffusionUniforms = UniformBufferUtils.createDiffusionUniforms(
        TIMESTEP,
        DIFFUSION_FACTOR
      );
      const diffusionBuffer = this.createUniformBuffer(
        diffusionUniforms,
        "Diffusion"
      );

      const diffusionPassEncoder = commandEncoder.beginComputePass({
        label: "Diffusion Compute Pass",
      });
      this.diffusionPass.execute(
        diffusionPassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: diffusionBuffer,
        },
        WORKGROUP_COUNT,
        DIFFUSION_ITERATIONS
      );
      diffusionPassEncoder.end();
      // Note: Diffusion pass handles its own texture swapping

      // Step 3: Divergence
      const divergenceUniforms =
        UniformBufferUtils.createDivergenceUniforms(GRID_SCALE);
      const divergenceBuffer = this.createUniformBuffer(
        divergenceUniforms,
        "Divergence"
      );

      const divergencePassEncoder = commandEncoder.beginComputePass({
        label: "Divergence Compute Pass",
      });
      this.divergencePass.execute(
        divergencePassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: divergenceBuffer,
        },
        WORKGROUP_COUNT
      );
      divergencePassEncoder.end();

      // Step 4: Pressure Projection
      const pressureUniforms =
        UniformBufferUtils.createPressureUniforms(GRID_SCALE);
      const pressureBuffer = this.createUniformBuffer(
        pressureUniforms,
        "Pressure"
      );

      const pressurePassEncoder = commandEncoder.beginComputePass({
        label: "Pressure Compute Pass",
      });
      this.pressurePass.execute(
        pressurePassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: pressureBuffer,
        },
        WORKGROUP_COUNT,
        PRESSURE_ITERATIONS
      );
      pressurePassEncoder.end();
      // Note: Pressure pass handles its own texture swapping

      // Step 5: Gradient Subtraction
      const gradientUniforms =
        UniformBufferUtils.createGradientSubtractionUniforms(GRID_SCALE);
      const gradientBuffer = this.createUniformBuffer(
        gradientUniforms,
        "Gradient"
      );

      const gradientPassEncoder = commandEncoder.beginComputePass({
        label: "Gradient Subtraction Compute Pass",
      });
      this.gradientSubtractionPass.execute(
        gradientPassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: gradientBuffer,
        },
        WORKGROUP_COUNT
      );
      gradientPassEncoder.end();
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
      {
        textureManager: this.textureManager,
        sampler: this.resources.device.createSampler({
          label: "Render Sampler",
        }),
      }
    );

    renderPassEncoder.end();
    this.resources.device.queue.submit([commandEncoder.finish()]);
  }

  public reset() {
    if (!this.resources || !this.textureManager || !this.isInitialized) {
      throw new WebGPUError(
        "Could not reset simulation: Resources not initialized",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    // Only reinitialize the velocity field
    initializeTextures(this.textureManager, this.resources.device);

    // Force a render
    this.step({ renderOnly: true });
  }

  private createUniformBuffer(data: Float32Array, label: string): GPUBuffer {
    const buffer = this.resources!.device.createBuffer({
      label: `${label} Uniform Buffer`,
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.resources!.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
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
      if (!canvasRef.current || !!smokeSimulation.current || isInitialized) {
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
  }, [isInitialized]);

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
      smokeSimulation.current.reset();
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
