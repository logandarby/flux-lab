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
import { RenderPass, ShaderMode } from "./utils/RenderPass";
import SimulationControls from "./components/ui/SimulationControls";
import {
  VelocityAdvectionPass,
  DiffusionPass,
  DivergencePass,
  GradientSubtractionPass,
  PressurePass,
  UniformBufferUtils,
  SmokeAdvectionPasss,
  SmokeDiffusionPass,
  BoundaryConditionsPass,
  BoundaryType,
} from "./passes/SimulationPasses";
import { usePersistedState } from "./utils/localStorage.utils";

// Configuration types for persistent settings
export interface SmokeSimulationConfig {
  shaderMode: ShaderMode;
  texture: SmokeTextureID;
}

// Constants

const CANVAS_WIDTH = 2 ** 9;
const CANVAS_HEIGHT = CANVAS_WIDTH;
const GRID_SIZE = {
  width: 2 ** 9,
  height: 2 ** 9,
};
const GRID_SCALE = 0.8;
const WORKGROUP_SIZE = 16;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE.width / WORKGROUP_SIZE);
// const VISCOSITY = 1;

const DIFFUSION_FACTOR = 10; // Smaller = slower diffusion
const VELOCITY_ADVECTION = 10; // Smaller = slower velocity advection
const SMOKE_ADVECTION = 10;
const SMOKE_DIFFUSION = 10;

const TIMESTEP = 1.0 / 30.0;
const DIFFUSION_ITERATIONS = 20;
const PRESSURE_ITERATIONS = 100;
const SMOKE_PARTICLE_DIMENSIONS = {
  width: GRID_SIZE.width,
  height: GRID_SIZE.height,
}; // TODO: Want this to work with arbitrary particle dimensions with advection

const INIT_VELOCITY = [-40, -40]; // TEMP
const INIT_SMOKE_DENSITY = 1.0;
const INIT_VELOCITY_SQUARE_SIZE = 2 ** 6;

// Utils

export type SmokeTextureID =
  | "divergence"
  | "velocity"
  | "pressure"
  | "smokeDensity"
  | "smokeParticlePosition";

function initializeTextures(
  textureManager: TextureManager<SmokeTextureID>,
  device: GPUDevice
): void {
  // Init Velocity
  const velocityTexture = textureManager.getCurrentTexture("velocity");
  // prettier-ignore
  const velocityData = Array(GRID_SIZE.width * GRID_SIZE.height).fill([0.0, 0.0]);
  const x = GRID_SIZE.width / 2;
  const y = GRID_SIZE.height / 2;
  const size = INIT_VELOCITY_SQUARE_SIZE;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      velocityData[x + i + GRID_SIZE.width * (y + j)] = INIT_VELOCITY;
    }
  }
  device.queue.writeTexture(
    { texture: velocityTexture },
    new Float32Array(velocityData.flat()),
    {
      bytesPerRow: GRID_SIZE.width * 2 * 4, // 2 channels × 4 bytes per 32-bit float
    },
    { ...GRID_SIZE }
  );

  // Init smoke density texture
  const smokeDensityTexture = textureManager.getCurrentTexture("smokeDensity");
  const initSmokeDensityData = Array(GRID_SIZE.width * GRID_SIZE.height).fill(
    0
  );
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      initSmokeDensityData[x + i + GRID_SIZE.width * (y + j)] =
        INIT_SMOKE_DENSITY;
    }
  }
  device.queue.writeTexture(
    { texture: smokeDensityTexture },
    new Float32Array(initSmokeDensityData),
    {
      bytesPerRow: GRID_SIZE.width * 4,
    },
    { ...GRID_SIZE }
  );

  // Init Smoke particle locations
  const smokeParticlePositionsTexture = textureManager.getCurrentTexture(
    "smokeParticlePosition"
  );
  const initSmokePositionsData = Array(
    SMOKE_PARTICLE_DIMENSIONS.width * SMOKE_PARTICLE_DIMENSIONS.height
  ).fill([0, 0]);
  for (let x = 0; x < SMOKE_PARTICLE_DIMENSIONS.width; x++) {
    for (let y = 0; y < SMOKE_PARTICLE_DIMENSIONS.height; y++) {
      initSmokePositionsData[x + y * SMOKE_PARTICLE_DIMENSIONS.width] = [
        (x / SMOKE_PARTICLE_DIMENSIONS.width) * GRID_SIZE.width,
        (y / SMOKE_PARTICLE_DIMENSIONS.height) * GRID_SIZE.height,
      ];
    }
  }
  device.queue.writeTexture(
    { texture: smokeParticlePositionsTexture },
    new Float32Array(initSmokePositionsData.flat()),
    {
      bytesPerRow: SMOKE_PARTICLE_DIMENSIONS.width * 4 * 2,
    },
    { ...SMOKE_PARTICLE_DIMENSIONS }
  );

  // Init pressure texture to all zeros
  const pressureTexture = textureManager.getCurrentTexture("pressure");
  const pressureBackTexture = textureManager.getBackTexture("pressure");
  const initPressureData = Array(GRID_SIZE.width * GRID_SIZE.height).fill(0.0);

  // Initialize both front and back pressure textures
  device.queue.writeTexture(
    { texture: pressureTexture },
    new Float32Array(initPressureData),
    {
      bytesPerRow: GRID_SIZE.width * 4, // 1 channel × 4 bytes per 32-bit float
    },
    { ...GRID_SIZE }
  );

  device.queue.writeTexture(
    { texture: pressureBackTexture },
    new Float32Array(initPressureData),
    {
      bytesPerRow: GRID_SIZE.width * 4,
    },
    { ...GRID_SIZE }
  );
}

class SmokeSimulation {
  private resources: WebGPUResources | null = null;
  private textureManager: TextureManager<SmokeTextureID> | null = null;
  private advectionPass: VelocityAdvectionPass | null = null;
  private smokeAdvectionPass: SmokeAdvectionPasss | null = null;
  private smokeDiffusionPass: SmokeDiffusionPass | null = null;
  private diffusionPass: DiffusionPass | null = null;
  private divergencePass: DivergencePass | null = null;
  private pressurePass: PressurePass | null = null;
  private gradientSubtractionPass: GradientSubtractionPass | null = null;
  private boundaryConditionsPass: BoundaryConditionsPass | null = null;
  private renderingPass: RenderPass<SmokeTextureID> | null = null;
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
      size: GRID_SIZE,
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create divergence texture
    this.textureManager.createTexture("divergence", {
      label: "Divergence texture",
      size: GRID_SIZE,
      format: "r32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create pressure texture (ping-pong)
    this.textureManager.createPingPongTexture("pressure", {
      label: "Pressure Texture",
      size: GRID_SIZE,
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Smoke particle texture (positions)
    this.textureManager.createPingPongTexture("smokeParticlePosition", {
      label: "Smoke Position Texture",
      size: SMOKE_PARTICLE_DIMENSIONS,
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Smoke density texture
    this.textureManager.createPingPongTexture("smokeDensity", {
      label: "Smoke Density Texture",
      size: GRID_SIZE,
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    initializeTextures(this.textureManager, this.resources.device);

    // Initialize simulation passes
    this.advectionPass = new VelocityAdvectionPass(
      this.resources.device,
      WORKGROUP_SIZE
    );
    this.smokeAdvectionPass = new SmokeAdvectionPasss(
      this.resources.device,
      WORKGROUP_SIZE
    );
    this.smokeDiffusionPass = new SmokeDiffusionPass(
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
    this.boundaryConditionsPass = new BoundaryConditionsPass(
      this.resources.device,
      WORKGROUP_SIZE
    );

    // Create rendering pass
    const textureShaderModule = this.resources.device.createShaderModule({
      label: "Texture Shader",
      code: textureShader,
    });
    this.renderingPass = new RenderPass<SmokeTextureID>(
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

    this.isInitialized = true;

    // Render the initial state
    this.step({ renderOnly: true });
  }

  public step({
    renderOnly = false,
    shaderMode = ShaderMode.DENSITY,
    texture = "smokeDensity",
  }: {
    renderOnly?: boolean;
    shaderMode?: ShaderMode;
    texture?: SmokeTextureID;
  } = {}) {
    if (
      !this.resources ||
      !this.textureManager ||
      !this.advectionPass ||
      !this.diffusionPass ||
      !this.divergencePass ||
      !this.pressurePass ||
      !this.gradientSubtractionPass ||
      !this.smokeAdvectionPass ||
      !this.smokeDiffusionPass ||
      !this.boundaryConditionsPass ||
      !this.renderingPass ||
      !this.isInitialized
    ) {
      throw new WebGPUError(
        "Could not step simulation: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    if (!renderOnly) {
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
      this.diffusionPass.executeIterations(
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
      this.pressurePass.executeIterations(
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

      // Step 4.5: Pressure Boundary Conditions (Neumann)
      const pressureBoundaryBuffer = this.createUniformBuffer(
        UniformBufferUtils.createBoundaryUniforms(BoundaryType.SCALAR_NEUMANN),
        "Pressure Boundary"
      );
      const pressureBoundaryPassEncoder = commandEncoder.beginComputePass({
        label: "Pressure Boundary Conditions Compute Pass",
      });
      this.boundaryConditionsPass.executeForTexture(
        pressureBoundaryPassEncoder,
        "pressure",
        {
          textureManager: this.textureManager,
          uniformBuffer: pressureBoundaryBuffer,
        },
        WORKGROUP_COUNT
      );
      pressureBoundaryPassEncoder.end();
      this.textureManager.swap("pressure");

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

      // Step 6: Velocity Boundary Conditions (No-slip)
      const velocityBoundaryBuffer = this.createUniformBuffer(
        UniformBufferUtils.createBoundaryUniforms(
          BoundaryType.NO_SLIP_VELOCITY
        ),
        "Velocity Boundary"
      );
      const velocityBoundaryPassEncoder = commandEncoder.beginComputePass({
        label: "Velocity Boundary Conditions Compute Pass",
      });
      this.boundaryConditionsPass.executeForTexture(
        velocityBoundaryPassEncoder,
        "velocity",
        {
          textureManager: this.textureManager,
          uniformBuffer: velocityBoundaryBuffer,
        },
        WORKGROUP_COUNT
      );
      velocityBoundaryPassEncoder.end();
      this.textureManager.swap("velocity");

      // Smoke Advection Pass
      const smokeAdvectionBuffer = this.createUniformBuffer(
        UniformBufferUtils.createAdvectionUniforms(TIMESTEP, SMOKE_ADVECTION),
        "Smoke Advection UBO"
      );
      const smokeDensityPassEncoder = commandEncoder.beginComputePass({
        label: "Smoke Density Advection Compute Pass",
      });
      this.smokeAdvectionPass.execute(
        smokeDensityPassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: smokeAdvectionBuffer,
        },
        WORKGROUP_COUNT
      );
      smokeDensityPassEncoder.end();
      this.textureManager.swap("smokeDensity");

      // Smoke Density Boundary Conditions (Neumann)
      const smokeBoundaryBuffer = this.createUniformBuffer(
        UniformBufferUtils.createBoundaryUniforms(BoundaryType.SCALAR_NEUMANN),
        "Smoke Boundary"
      );
      const smokeBoundaryPassEncoder = commandEncoder.beginComputePass({
        label: "Smoke Boundary Conditions Compute Pass",
      });
      this.boundaryConditionsPass.executeForTexture(
        smokeBoundaryPassEncoder,
        "smokeDensity",
        {
          textureManager: this.textureManager,
          uniformBuffer: smokeBoundaryBuffer,
        },
        WORKGROUP_COUNT
      );
      smokeBoundaryPassEncoder.end();
      this.textureManager.swap("smokeDensity");
    }

    const smokeDiffusionBuffer = this.createUniformBuffer(
      UniformBufferUtils.createDiffusionUniforms(TIMESTEP, SMOKE_DIFFUSION),
      "Smoke Diffusion UBO"
    );
    const smokeDiffusionPassEncoder = commandEncoder.beginComputePass({
      label: "Smoke Density Diffusion Compute Pass",
    });
    this.smokeDiffusionPass.executeIterations(
      smokeDiffusionPassEncoder,
      {
        textureManager: this.textureManager,
        uniformBuffer: smokeDiffusionBuffer,
      },
      WORKGROUP_COUNT,
      DIFFUSION_ITERATIONS
    );
    smokeDiffusionPassEncoder.end();
    this.textureManager.swap("smokeDensity");

    // Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass({
      label: "Texture Render Pass",
      colorAttachments: [
        {
          view: this.resources.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
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
        shaderMode: shaderMode,
        texture: texture,
      }
    );

    renderPassEncoder.end();
    this.resources.device.queue.submit([commandEncoder.finish()]);
  }

  public reset(
    shaderMode: ShaderMode = ShaderMode.DENSITY,
    texture: SmokeTextureID = "smokeDensity"
  ) {
    if (!this.resources || !this.textureManager || !this.isInitialized) {
      throw new WebGPUError(
        "Could not reset simulation: Resources not initialized",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    // Only reinitialize the velocity field
    initializeTextures(this.textureManager, this.resources.device);

    // Force a render
    this.step({ renderOnly: true, shaderMode, texture });
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
  const [selectedMode, setSelectedMode] = usePersistedState<ShaderMode>(
    "smoke-simulation-shader-mode",
    ShaderMode.DENSITY
  );
  const [selectedTexture, setSelectedTexture] =
    usePersistedState<SmokeTextureID>(
      "smoke-simulation-texture",
      "smokeDensity"
    );

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
          smokeSimulation.current.step({
            shaderMode: selectedMode,
            texture: selectedTexture,
          });
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
  }, [isPlaying, isInitialized, selectedMode, selectedTexture]);

  // Re-render when visualization mode changes (if not playing)
  useEffect(() => {
    if (isInitialized && !isPlaying && smokeSimulation.current) {
      try {
        smokeSimulation.current.step({
          renderOnly: true,
          shaderMode: selectedMode,
          texture: selectedTexture,
        });
      } catch (error) {
        console.error("Failed to update visualization:", error);
      }
    }
  }, [selectedMode, selectedTexture, isInitialized, isPlaying]);

  const handleStep = useCallback(() => {
    if (smokeSimulation.current && isInitialized) {
      try {
        smokeSimulation.current.step({
          shaderMode: selectedMode,
          texture: selectedTexture,
        });
      } catch (error) {
        console.error("Failed to step simulation:", error);
      }
    }
  }, [isInitialized, selectedMode, selectedTexture]);

  const handleRestart = useCallback(async () => {
    if (!smokeSimulation.current || !isInitialized) {
      return;
    }

    try {
      smokeSimulation.current.reset(selectedMode, selectedTexture);
    } catch (error) {
      console.error("Failed to restart simulation:", error);
    }
  }, [isInitialized, selectedMode, selectedTexture]);

  // Visualization mode options
  const visualizationModes = [
    {
      value: ShaderMode.DENSITY,
      label: "Smoke Density",
      texture: "smokeDensity" as SmokeTextureID,
    },
    {
      value: ShaderMode.VELOCITY,
      label: "Velocity Field",
      texture: "velocity" as SmokeTextureID,
    },
    {
      value: ShaderMode.PRESSURE,
      label: "Pressure Field",
      texture: "pressure" as SmokeTextureID,
    },
  ];

  const handleModeChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const mode = parseInt(event.target.value) as ShaderMode;
      const modeOption = visualizationModes.find((m) => m.value === mode);
      if (modeOption) {
        setSelectedMode(mode);
        setSelectedTexture(modeOption.texture);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setSelectedMode, setSelectedTexture]
  );

  // Custom visualization mode dropdown
  const visualizationControl = (
    <div className="w-full">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Visualization Mode
      </label>
      <select
        value={selectedMode}
        onChange={handleModeChange}
        disabled={!isInitialized}
        className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
      >
        {visualizationModes.map((mode) => (
          <option key={mode.value} value={mode.value}>
            {mode.label}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="p-5">
      <div className="max-w-6xl mx-auto flex flex-col items-center gap-6">
        {/* Title and Description */}
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            Smoke Simulation
          </h2>
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
            customControls={visualizationControl}
          />
        </div>
      </div>
    </div>
  );
}

export default SmokeSimulationComponent;
