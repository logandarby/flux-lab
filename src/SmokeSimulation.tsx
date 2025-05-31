import textureShader from "./shaders/textureShader.wgsl?raw";
import {
  initializeWebGPU,
  WebGPUError,
  WebGPUErrorCode,
  type WebGPUResources,
} from "./utils/webgpu.utils";
import { TextureManager } from "./utils/TextureManager";
import { RenderPass, ShaderMode } from "./utils/RenderPass";
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
  AddSmokePass,
  AddVelocityPass,
  SmokeDissipationPass,
  VelocityDissipationPass,
} from "./passes/SimulationPasses";

// Configuration types for persistent settings
export interface SmokeSimulationConfig {
  shaderMode: ShaderMode;
  texture: SmokeTextureID;
}

// Constants

// Alternate setting

export const GRID_SIZE = {
  width: 2 ** 9,
  height: 2 ** 9,
};
const GRID_SCALE = 0.8;
const WORKGROUP_SIZE = 16;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE.width / WORKGROUP_SIZE);
// const VISCOSITY = 1;

const DIFFUSION_FACTOR = 10; // Smaller = slower diffusion
const VELOCITY_ADVECTION = 10; // Smaller = slower velocity advection
const SMOKE_ADVECTION = 20;
const SMOKE_DIFFUSION = 1;
const SMOKE_DISSIPATION_FACTOR = 0.99; // Multiplication factor for smoke density each frame
const VELOCITY_DISSIPATION_FACTOR = 0.995; // Similary for velocity magnitude

// Interaction constants
const INTERACTION_RADIUS = 20; // Radius of effect when adding smoke/velocity
const SMOKE_INTENSITY = 0.8; // Intensity of smoke when clicking
const VELOCITY_INTENSITY = 2.0; // Intensity multiplier for velocity when dragging (reduced significantly)

const TIMESTEP = 1.0 / 30.0;
const DIFFUSION_ITERATIONS = 40;
const PRESSURE_ITERATIONS = 100;
const SMOKE_PARTICLE_DIMENSIONS = {
  width: GRID_SIZE.width,
  height: GRID_SIZE.height,
};

// TODO: Refactor into its own file-- maybe let them be modified.
// export const GRID_SIZE = {
//   width: 2 ** 9,
//   height: 2 ** 9,
// };
// const GRID_SCALE = 0.8;
// const WORKGROUP_SIZE = 16;
// const WORKGROUP_COUNT = Math.ceil(GRID_SIZE.width / WORKGROUP_SIZE);
// // const VISCOSITY = 1;

// const DIFFUSION_FACTOR = 1; // Smaller = slower diffusion
// const VELOCITY_ADVECTION = 10; // Smaller = slower velocity advection
// const SMOKE_ADVECTION = 20;
// const SMOKE_DIFFUSION = 1;

// // Interaction constants
// const INTERACTION_RADIUS = 20; // Radius of splat when adding smoke/velocity
// const SMOKE_INTENSITY = 0.8; // Intensity of smoke when clicking
// const VELOCITY_INTENSITY = 4.0; // Intensity multiplier for velocity when dragging

// // Dissipation constant - higher value means slower dissipation (closer to 1.0)
// const SMOKE_DISSIPATION_FACTOR = 0.99; // Multiplication factor for smoke density each frame
// const VELOCITY_DISSIPATION_FACTOR = 0.995; // Similary for velocity magnitude

// const TIMESTEP = 1.0 / 60.0;
// const DIFFUSION_ITERATIONS = 40;
// const PRESSURE_ITERATIONS = 100;
// const SMOKE_PARTICLE_DIMENSIONS = {
//   width: GRID_SIZE.width,
//   height: GRID_SIZE.height,
// }; // TODO: Want this to work with arbitrary particle dimensions with advection

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
  const velocityData = Array(GRID_SIZE.width * GRID_SIZE.height).fill([
    0.0, 0.0,
  ]);
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
  device.queue.writeTexture(
    { texture: smokeDensityTexture },
    new Float32Array(Array(GRID_SIZE.width * GRID_SIZE.height).fill(0)),
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
  private smokeDissipationPass: SmokeDissipationPass | null = null;
  private velocityDissipationPass: VelocityDissipationPass | null = null;
  private diffusionPass: DiffusionPass | null = null;
  private divergencePass: DivergencePass | null = null;
  private pressurePass: PressurePass | null = null;
  private gradientSubtractionPass: GradientSubtractionPass | null = null;
  private boundaryConditionsPass: BoundaryConditionsPass | null = null;
  private addSmokePass: AddSmokePass | null = null;
  private addVelocityPass: AddVelocityPass | null = null;
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
    this.smokeDissipationPass = new SmokeDissipationPass(
      this.resources.device,
      WORKGROUP_SIZE
    );
    this.velocityDissipationPass = new VelocityDissipationPass(
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

    // Initialize user interaction passes
    this.addSmokePass = new AddSmokePass(this.resources.device, WORKGROUP_SIZE);
    this.addVelocityPass = new AddVelocityPass(
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
      !this.smokeDissipationPass ||
      !this.velocityDissipationPass ||
      !this.boundaryConditionsPass ||
      !this.addSmokePass ||
      !this.addVelocityPass ||
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

      // Velocity Dissipation Pass - Apply dissipation to gradually reduce velocity
      const velocityDissipationBuffer = this.createUniformBuffer(
        UniformBufferUtils.createDissipationUniforms(
          VELOCITY_DISSIPATION_FACTOR
        ),
        "Velocity Dissipation UBO"
      );
      const velocityDissipationPassEncoder = commandEncoder.beginComputePass({
        label: "Velocity Dissipation Compute Pass",
      });
      this.velocityDissipationPass.execute(
        velocityDissipationPassEncoder,
        {
          textureManager: this.textureManager,
          uniformBuffer: velocityDissipationBuffer,
        },
        WORKGROUP_COUNT
      );
      velocityDissipationPassEncoder.end();
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

    // Smoke Dissipation Pass
    const smokeDissipationBuffer = this.createUniformBuffer(
      UniformBufferUtils.createDissipationUniforms(SMOKE_DISSIPATION_FACTOR),
      "Smoke Dissipation UBO"
    );
    const smokeDissipationPassEncoder = commandEncoder.beginComputePass({
      label: "Smoke Density Dissipation Compute Pass",
    });
    this.smokeDissipationPass.execute(
      smokeDissipationPassEncoder,
      {
        textureManager: this.textureManager,
        uniformBuffer: smokeDissipationBuffer,
      },
      WORKGROUP_COUNT
    );
    smokeDissipationPassEncoder.end();
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
    this.renderingPass.writeToUniformBuffer({
      shaderMode,
      noise: {
        mean: 0,
        stddev: 0.05,
        offsets: [Math.random(), Math.random(), Math.random()],
      },
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

  public addSmoke(x: number, y: number) {
    if (
      !this.resources ||
      !this.textureManager ||
      !this.addSmokePass ||
      !this.isInitialized
    ) {
      return;
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    const addSmokeUniforms = UniformBufferUtils.createAddSmokeUniforms(
      x,
      y,
      INTERACTION_RADIUS,
      SMOKE_INTENSITY
    );
    const addSmokeBuffer = this.createUniformBuffer(
      addSmokeUniforms,
      "Add Smoke"
    );

    const addSmokePassEncoder = commandEncoder.beginComputePass({
      label: "Add Smoke Compute Pass",
    });
    this.addSmokePass.execute(
      addSmokePassEncoder,
      {
        textureManager: this.textureManager,
        uniformBuffer: addSmokeBuffer,
      },
      WORKGROUP_COUNT
    );
    addSmokePassEncoder.end();
    this.textureManager.swap("smokeDensity");

    this.resources.device.queue.submit([commandEncoder.finish()]);
  }

  public addVelocity(
    x: number,
    y: number,
    velocityX: number,
    velocityY: number
  ) {
    if (
      !this.resources ||
      !this.textureManager ||
      !this.addVelocityPass ||
      !this.isInitialized
    ) {
      return;
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    const addVelocityUniforms = UniformBufferUtils.createAddVelocityUniforms(
      x,
      y,
      velocityX,
      velocityY,
      INTERACTION_RADIUS,
      VELOCITY_INTENSITY
    );
    const addVelocityBuffer = this.createUniformBuffer(
      addVelocityUniforms,
      "Add Velocity"
    );

    const addVelocityPassEncoder = commandEncoder.beginComputePass({
      label: "Add Velocity Compute Pass",
    });
    this.addVelocityPass.execute(
      addVelocityPassEncoder,
      {
        textureManager: this.textureManager,
        uniformBuffer: addVelocityBuffer,
      },
      WORKGROUP_COUNT
    );
    addVelocityPassEncoder.end();
    this.textureManager.swap("velocity");

    this.resources.device.queue.submit([commandEncoder.finish()]);
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

export default SmokeSimulation;
