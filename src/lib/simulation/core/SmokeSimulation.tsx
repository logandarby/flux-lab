import React from "react";
import {
  initializeWebGPU,
  WebGPUError,
  WebGPUErrorCode,
  type WebGPUResources,
  GPUTimer,
} from "@/shared/webgpu/webgpu.utils";
import { TextureManager } from "@/shared/webgpu/TextureManager";
import { RenderPass, ShaderMode } from "@/shared/webgpu/RenderPass";
import { PerformanceTracker, type PerformanceMetrics } from "@/lib/performance";
import {
  VelocityAdvectionPass,
  DiffusionPass,
  DivergencePass,
  GradientSubtractionPass,
  PressurePass,
  SmokeAdvectionPasss,
  SmokeDiffusionPass,
  BoundaryConditionsPass,
  BoundaryType,
  AddSmokePass,
  AddVelocityPass,
  SmokeDissipationPass,
  VelocityDissipationPass,
  BindLayoutManager,
  AdvectParticlesPass,
} from "./SimulationPasses";
import {
  AdvectionUniforms,
  DiffusionUniforms,
  DivergenceUniforms,
  PressureUniforms,
  GradientSubtractionUniforms,
  BoundaryUniforms,
  AddSmokeUniforms,
  AddVelocityUniforms,
  DissipationUniforms,
  AdvectParticlesUniforms,
} from "@/shared/webgpu/UniformManager";
import {
  type SmokeTextureID,
  type SimulationStepConfig,
  DEFAULT_VELOCITY_CONTROLS,
  DEFAULT_SMOKE_CONTROLS,
  type SmokeSimulationConfig,
  type SimulationConstants,
} from "./types";
import { SIMULATION_CONSTANTS } from "./constants";
import { wgsl } from "@/lib/preprocessor/core/wgsl";
import { SHADERS } from "../shaders";
import { FLOAT_BYTES } from "./constants";
import { createScalarField2D, createVectorField2D } from "../utils";
import type { ScalarField2D, VectorField2D } from "../utils";

export interface SmokeTextureExports {
  smokeDensity: ScalarField2D;
  velocity: VectorField2D;
  pressure: ScalarField2D;
}

interface ISmokeSimulation {
  initialize(canvas: React.RefObject<HTMLCanvasElement>): void;
  destroy(): void;

  step(config: SmokeSimulationConfig): void;
  reset(): void;

  addSmoke(x: number, y: number): void;
  addVelocity(x: number, y: number, velocityX: number, velocityY: number): void;

  sampleTextures(downSample: number): Promise<SmokeTextureExports>;

  getPerformanceMetrics(): PerformanceMetrics;
  setSmokeColor(color: [number, number, number]): void;
}

// Deep merge utility function for SimulationConstants
function mergeSimulationConstants(
  defaults: SimulationConstants,
  overrides: Partial<SimulationConstants>
): SimulationConstants {
  const mergedGrid = {
    ...defaults.grid,
    ...overrides.grid,
  };

  const mergedCompute = {
    ...defaults.compute,
    ...overrides.compute,
  };

  // Compute workgroupCount based on the final merged values
  const finalCompute = {
    ...mergedCompute,
    workgroupCount: Math.ceil(
      mergedGrid.size.width / mergedCompute.workgroupSize
    ),
  };

  return {
    grid: mergedGrid,
    compute: finalCompute,
    physics: {
      ...defaults.physics,
      ...overrides.physics,
    },
    interaction: {
      ...defaults.interaction,
      ...overrides.interaction,
    },
    iterations: {
      ...defaults.iterations,
      ...overrides.iterations,
    },
    particles: {
      ...defaults.particles,
      ...overrides.particles,
    },
  };
}

class SmokeSimulation implements ISmokeSimulation {
  private resources: WebGPUResources | null = null;
  private textureManager: TextureManager<SmokeTextureID> | null = null;
  private performanceTracker = new PerformanceTracker();
  private gpuTimer: GPUTimer | null = null;
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
  private advectParticlesPass: AdvectParticlesPass | null = null;
  private isInitialized = false;
  private lastFrameTime: number = performance.now();
  private config: SimulationConstants;
  private smokeColor: [number, number, number] = [0.65, 0.35, 0.85]; // Default purple color

  constructor(customConfig?: Partial<SimulationConstants>) {
    this.config = mergeSimulationConstants(
      SIMULATION_CONSTANTS,
      customConfig || {}
    );
  }

  public setSmokeColor(color: [number, number, number]): void {
    this.smokeColor = color;
  }

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

    // Initialize GPU timer and performance tracking
    this.gpuTimer = new GPUTimer(
      this.resources.device,
      this.resources.canTimestamp
    );
    this.performanceTracker.setGpuSupported(this.resources.canTimestamp);

    // Initialize texture manager
    this.textureManager = new TextureManager<SmokeTextureID>(
      this.resources.device
    );

    // Create velocity texture (ping-pong)
    this.textureManager.createPingPongTexture("velocity", {
      label: "Velocity Texture",
      size: this.config.grid.size,
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });

    // Create divergence texture
    this.textureManager.createTexture("divergence", {
      label: "Divergence texture",
      size: this.config.grid.size,
      format: "r32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create pressure texture (ping-pong)
    this.textureManager.createPingPongTexture("pressure", {
      label: "Pressure Texture",
      size: this.config.grid.size,
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });

    // Smoke particle texture (positions)
    this.textureManager.createPingPongTexture("smokeParticlePosition", {
      label: "Smoke Position Texture",
      size: this.config.particles.smokeDimensions,
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Smoke density texture
    this.textureManager.createPingPongTexture("smokeDensity", {
      label: "Smoke Density Texture",
      size: this.config.grid.size,
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });

    initializeTextures(this.textureManager, this.resources.device, this.config);

    // Initialize simulation passes
    this.advectionPass = new VelocityAdvectionPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.smokeAdvectionPass = new SmokeAdvectionPasss(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.smokeDiffusionPass = new SmokeDiffusionPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.smokeDissipationPass = new SmokeDissipationPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.velocityDissipationPass = new VelocityDissipationPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.diffusionPass = new DiffusionPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.divergencePass = new DivergencePass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.pressurePass = new PressurePass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.gradientSubtractionPass = new GradientSubtractionPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.boundaryConditionsPass = new BoundaryConditionsPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.advectParticlesPass = new AdvectParticlesPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );

    // Initialize user interaction passes
    this.addSmokePass = new AddSmokePass(
      this.resources.device,
      this.config.compute.workgroupSize
    );
    this.addVelocityPass = new AddVelocityPass(
      this.resources.device,
      this.config.compute.workgroupSize
    );

    // Create rendering pass
    const textureShaderModule = this.resources.device.createShaderModule({
      label: "Texture Shader",
      code: wgsl(SHADERS.TEXTURE),
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

  public step(config: SimulationStepConfig = {}) {
    const jsStartTime = performance.now();
    this.performanceTracker.recordFrame();

    // Calculate real-time timestep
    const currentTime = performance.now();
    const realTimeTimestep = Math.min(
      (currentTime - this.lastFrameTime) / 1000,
      this.config.physics.maxTimestep
    ); // Cap at maxTimestep to prevent instability during frame drops
    this.lastFrameTime = currentTime;

    // Record timestep for performance tracking
    this.performanceTracker.recordTimestep(realTimeTimestep);

    // Destructure with defaults
    const {
      renderOnly = false,
      shaderMode = ShaderMode.DENSITY,
      texture = "smokeDensity",
      velocity: velocityConfig = {},
      smoke: smokeConfig = {},
    } = config;

    // Merge with defaults
    const velocityControls = {
      ...DEFAULT_VELOCITY_CONTROLS,
      ...velocityConfig,
    };
    const smokeControls = { ...DEFAULT_SMOKE_CONTROLS, ...smokeConfig };

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
      !this.advectParticlesPass ||
      !this.isInitialized
    ) {
      throw new WebGPUError(
        "Could not step simulation: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    // Compute Pass -- Step simulation forward
    if (!renderOnly) {
      this.executeComputePasses(
        commandEncoder,
        velocityControls,
        smokeControls,
        realTimeTimestep
      );
    }

    // Render Pass
    // Ensure canvas context is properly configured with current device
    this.resources.context.configure({
      device: this.resources.device,
      format: this.resources.canvasFormat,
    });

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
      ...(this.gpuTimer?.getTimestampWrites() && {
        timestampWrites: this.gpuTimer.getTimestampWrites(),
      }),
    });
    this.renderingPass.writeToUniformBuffer({
      shaderMode,
      noise: {
        mean: 0,
        stddev: 0.05,
        offsets: [Math.random(), Math.random(), Math.random()],
      },
      smokeColor: this.smokeColor,
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

    // Resolve GPU timing
    if (this.gpuTimer) {
      this.gpuTimer.resolveTimestamps(commandEncoder);
    }

    this.resources.device.queue.submit([commandEncoder.finish()]);

    // Read GPU timing results asynchronously
    if (this.gpuTimer) {
      this.gpuTimer.readResults().then(() => {
        this.performanceTracker.recordGPU(this.gpuTimer!.getLastGpuTime());
      });
    }

    // Record JS performance time
    const jsElapsed = performance.now() - jsStartTime;
    this.performanceTracker.recordJS(jsElapsed);
  }

  private executeComputePasses(
    commandEncoder: GPUCommandEncoder,
    velocityControls: typeof DEFAULT_VELOCITY_CONTROLS,
    smokeControls: typeof DEFAULT_SMOKE_CONTROLS,
    realTimeTimestep: number
  ) {
    /**
     * NOTE: It's safe to do all compute operations in one pass,
     * because WebGPU limits all `dispatchWorkgroup` calls to their own usage scope. This means we avoid the possibility of any
     * RAW, WAR, or WAW hazards.
     * More information at https://www.w3.org/TR/webgpu/#programming-model-synchronization
     */

    const computePassEncoder = commandEncoder.beginComputePass({
      label: "Compute Pass",
    });

    if (velocityControls.enableAdvection) {
      const advectionUniforms = new AdvectionUniforms(
        realTimeTimestep,
        this.config.physics.velocityAdvection
      );
      this.advectionPass!.executeWithUniforms(
        computePassEncoder,
        advectionUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (velocityControls.enableDiffusion) {
      const diffusionUniforms = new DiffusionUniforms(
        realTimeTimestep,
        this.config.physics.diffusionFactor
      );
      this.diffusionPass!.executeIterations(
        computePassEncoder,
        diffusionUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount,
        this.config.iterations.diffusion
      );
    }

    if (velocityControls.enableDivergence) {
      const divergenceUniforms = new DivergenceUniforms(this.config.grid.scale);
      this.divergencePass!.executeWithUniforms(
        computePassEncoder,
        divergenceUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
    }

    if (velocityControls.enablePressureProjection) {
      const pressureUniforms = new PressureUniforms(this.config.grid.scale);
      this.pressurePass!.executeIterations(
        computePassEncoder,
        pressureUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount,
        this.config.iterations.pressure
      );
    }

    if (velocityControls.enablePressureBoundaryConditions) {
      const pressureBoundaryUniforms = new BoundaryUniforms(
        BoundaryType.SCALAR_NEUMANN
      );
      this.boundaryConditionsPass!.executeForTexture(
        computePassEncoder,
        "pressure",
        pressureBoundaryUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("pressure");
    }

    if (velocityControls.enableGradientSubtraction) {
      const gradientUniforms = new GradientSubtractionUniforms(
        this.config.grid.scale
      );
      this.gradientSubtractionPass!.executeWithUniforms(
        computePassEncoder,
        gradientUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (velocityControls.enableVelocityBoundaryConditions) {
      const velocityBoundaryUniforms = new BoundaryUniforms(
        BoundaryType.NO_SLIP_VELOCITY
      );
      this.boundaryConditionsPass!.executeForTexture(
        computePassEncoder,
        "velocity",
        velocityBoundaryUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (velocityControls.enableDissipation) {
      const velocityDissipationUniforms = new DissipationUniforms(
        this.config.physics.velocityDissipationFactor
      );
      this.velocityDissipationPass!.executeWithUniforms(
        computePassEncoder,
        velocityDissipationUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (smokeControls.enableAdvection) {
      const smokeAdvectionUniforms = new AdvectionUniforms(
        realTimeTimestep,
        this.config.physics.smokeAdvection
      );
      this.smokeAdvectionPass!.executeWithUniforms(
        computePassEncoder,
        smokeAdvectionUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("smokeDensity");
    }

    if (smokeControls.enableBoundaryConditions) {
      const smokeBoundaryUniforms = new BoundaryUniforms(
        BoundaryType.SCALAR_NEUMANN
      );
      this.boundaryConditionsPass!.executeForTexture(
        computePassEncoder,
        "smokeDensity",
        smokeBoundaryUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("smokeDensity");
    }

    if (smokeControls.enableDiffusion) {
      const smokeDiffusionUniforms = new DiffusionUniforms(
        realTimeTimestep,
        this.config.physics.smokeDiffusion
      );
      this.smokeDiffusionPass!.executeIterations(
        computePassEncoder,
        smokeDiffusionUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount,
        this.config.iterations.diffusion
      );
      this.textureManager!.swap("smokeDensity");
    }

    if (smokeControls.enableDissipation) {
      const smokeDissipationUniforms = new DissipationUniforms(
        this.config.physics.smokeDissipationFactor
      );
      this.smokeDissipationPass!.executeWithUniforms(
        computePassEncoder,
        smokeDissipationUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("smokeDensity");
    }

    if (smokeControls.enableAdvectParticles) {
      const advectParticlesUniforms = new AdvectParticlesUniforms(
        realTimeTimestep
      );
      this.advectParticlesPass!.executeWithUniforms(
        computePassEncoder,
        advectParticlesUniforms,
        this.textureManager!,
        this.config.compute.workgroupCount
      );
      this.textureManager!.swap("smokeParticlePosition");
    }

    computePassEncoder.end();
  }

  public getPerformanceMetrics() {
    return this.performanceTracker.getMetrics();
  }

  public getGridSize() {
    return this.config.grid.size;
  }

  public async sampleTextures(
    downSample: number = 1
  ): Promise<SmokeTextureExports> {
    if (!this.resources || !this.textureManager || !this.isInitialized) {
      throw new WebGPUError(
        "Could not sample textures: Resources not initialized",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const gridSize = this.config.grid.size;
    const sampledWidth = Math.floor(gridSize.width / downSample);
    const sampledHeight = Math.floor(gridSize.height / downSample);
    const sampledPixels = sampledWidth * sampledHeight;

    // Create staging buffers for reading texture data
    const smokeDensityBuffer = this.resources.device.createBuffer({
      label: "Smoke Density Staging Buffer",
      size: sampledPixels * FLOAT_BYTES, // 1 float32 per pixel
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const velocityBuffer = this.resources.device.createBuffer({
      label: "Velocity Staging Buffer",
      size: sampledPixels * FLOAT_BYTES * 2, // 2 float32s per pixel
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const pressureBuffer = this.resources.device.createBuffer({
      label: "Pressure Staging Buffer",
      size: sampledPixels * FLOAT_BYTES, // 1 float32 per pixel
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create command encoder for copying texture data
    const commandEncoder = this.resources.device.createCommandEncoder({
      label: "Texture Sampling Command Encoder",
    });

    // Copy texture data to staging buffers
    commandEncoder.copyTextureToBuffer(
      { texture: this.textureManager.getCurrentTexture("smokeDensity") },
      {
        buffer: smokeDensityBuffer,
        bytesPerRow: sampledWidth * FLOAT_BYTES,
      },
      { width: sampledWidth, height: sampledHeight }
    );

    commandEncoder.copyTextureToBuffer(
      { texture: this.textureManager.getCurrentTexture("velocity") },
      {
        buffer: velocityBuffer,
        bytesPerRow: sampledWidth * FLOAT_BYTES * 2,
      },
      { width: sampledWidth, height: sampledHeight }
    );

    commandEncoder.copyTextureToBuffer(
      { texture: this.textureManager.getCurrentTexture("pressure") },
      {
        buffer: pressureBuffer,
        bytesPerRow: sampledWidth * FLOAT_BYTES,
      },
      { width: sampledWidth, height: sampledHeight }
    );

    this.resources.device.queue.submit([commandEncoder.finish()]);
    await this.resources.device.queue.onSubmittedWorkDone();

    await Promise.all([
      smokeDensityBuffer.mapAsync(GPUMapMode.READ),
      velocityBuffer.mapAsync(GPUMapMode.READ),
      pressureBuffer.mapAsync(GPUMapMode.READ),
    ]);

    const smokeDensityData = new Float32Array(
      smokeDensityBuffer.getMappedRange()
    );
    const velocityRawData = new Float32Array(velocityBuffer.getMappedRange());
    const pressureData = new Float32Array(pressureBuffer.getMappedRange());

    // Convert velocity data to array of vec2 tuples
    const velocityData: Array<[number, number]> = [];
    for (let i = 0; i < velocityRawData.length; i += 2) {
      velocityData.push([velocityRawData[i], velocityRawData[i + 1]]);
    }

    // Create Array2D structures with proper dimensions and flipped Y coordinates
    const result: SmokeTextureExports = {
      smokeDensity: createScalarField2D(
        new Float32Array(smokeDensityData),
        gridSize.width,
        gridSize.height,
        downSample,
        true
      ),
      velocity: createVectorField2D(
        velocityData,
        gridSize.width,
        gridSize.height,
        downSample,
        true
      ),
      pressure: createScalarField2D(
        new Float32Array(pressureData),
        gridSize.width,
        gridSize.height,
        downSample,
        true
      ),
    };

    // Unmap and destroy staging buffers
    smokeDensityBuffer.unmap();
    velocityBuffer.unmap();
    pressureBuffer.unmap();
    smokeDensityBuffer.destroy();
    velocityBuffer.destroy();
    pressureBuffer.destroy();

    return result;
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

    // Reset frame time to prevent large timestep on first frame after reset
    this.lastFrameTime = performance.now();

    // Reinitialize textures
    initializeTextures(this.textureManager, this.resources.device, this.config);

    // Render initial state
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

    const addSmokeUniforms = new AddSmokeUniforms(
      x,
      y,
      this.config.interaction.radius,
      this.config.interaction.smokeIntensity
    );

    const addSmokePassEncoder = commandEncoder.beginComputePass({
      label: "Add Smoke Compute Pass",
    });
    this.addSmokePass.executeWithUniforms(
      addSmokePassEncoder,
      addSmokeUniforms,
      this.textureManager,
      this.config.compute.workgroupCount
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

    const addVelocityUniforms = new AddVelocityUniforms(
      x,
      y,
      velocityX,
      velocityY,
      this.config.interaction.radius,
      this.config.interaction.velocityIntensity
    );

    const addVelocityPassEncoder = commandEncoder.beginComputePass({
      label: "Add Velocity Compute Pass",
    });
    this.addVelocityPass.executeWithUniforms(
      addVelocityPassEncoder,
      addVelocityUniforms,
      this.textureManager,
      this.config.compute.workgroupCount
    );
    addVelocityPassEncoder.end();
    this.textureManager.swap("velocity");

    this.resources.device.queue.submit([commandEncoder.finish()]);
  }

  // Just cleanup heavy resources -- textures and uniform buffers
  public destroy(): void {
    if (!this.isInitialized) {
      return;
    }

    // Clean up heavy resources
    this.textureManager?.destroy(); // Multiple large textures
    this.renderingPass?.destroy(); // Uniform buffers
    this.gpuTimer?.destroy(); // GPU timing resources
    this.advectionPass?.destroy();
    this.smokeAdvectionPass?.destroy();
    this.smokeDiffusionPass?.destroy();
    this.smokeDissipationPass?.destroy();
    this.velocityDissipationPass?.destroy();
    this.diffusionPass?.destroy();
    this.divergencePass?.destroy();
    this.pressurePass?.destroy();
    this.gradientSubtractionPass?.destroy();
    this.boundaryConditionsPass?.destroy();
    this.addSmokePass?.destroy();
    this.addVelocityPass?.destroy();

    // Clean up device-specific caches
    if (this.resources?.device) {
      BindLayoutManager.destroyDevice(this.resources.device);
    }

    // Clear critical references
    this.resources = null;
    this.textureManager = null;
    this.renderingPass = null;

    console.log("SmokeSimulation destroyed");
    this.isInitialized = false;
  }
}

function initializeTextures(
  textureManager: TextureManager<SmokeTextureID>,
  device: GPUDevice,
  config: SimulationConstants
): void {
  const gridSize = config.grid.size;
  const particleSize = config.particles.smokeDimensions;
  const totalGridPixels = gridSize.width * gridSize.height;
  const totalParticlePixels = particleSize.width * particleSize.height;

  // Pre-allocate buffers to reduce memory allocations
  const velocityData = new Float32Array(totalGridPixels * 2); // 2 channels
  const smokeDensityData = new Float32Array(totalGridPixels); // 1 channel (already zeros)
  const pressureData = new Float32Array(totalGridPixels); // 1 channel (already zeros)
  const particlePositionData = new Float32Array(totalParticlePixels * 2); // 2 channels

  // Initialize particle positions efficiently
  for (let y = 0; y < particleSize.height; y++) {
    for (let x = 0; x < particleSize.width; x++) {
      const index = (x + y * particleSize.width) * 2;
      particlePositionData[index] = (x / particleSize.width) * gridSize.width;
      particlePositionData[index + 1] =
        (y / particleSize.height) * gridSize.height;
    }
  }

  // Write all textures
  const velocityTexture = textureManager.getCurrentTexture("velocity");
  device.queue.writeTexture(
    { texture: velocityTexture },
    velocityData,
    { bytesPerRow: gridSize.width * 2 * FLOAT_BYTES },
    gridSize
  );

  const smokeDensityTexture = textureManager.getCurrentTexture("smokeDensity");
  device.queue.writeTexture(
    { texture: smokeDensityTexture },
    smokeDensityData,
    { bytesPerRow: gridSize.width * FLOAT_BYTES },
    gridSize
  );

  const smokeParticlePositionsTexture = textureManager.getCurrentTexture(
    "smokeParticlePosition"
  );
  device.queue.writeTexture(
    { texture: smokeParticlePositionsTexture },
    particlePositionData,
    { bytesPerRow: particleSize.width * FLOAT_BYTES * 2 },
    particleSize
  );

  // Initialize pressure textures
  const pressureTexture = textureManager.getCurrentTexture("pressure");
  const pressureBackTexture = textureManager.getBackTexture("pressure");

  device.queue.writeTexture(
    { texture: pressureTexture },
    pressureData,
    { bytesPerRow: gridSize.width * FLOAT_BYTES },
    gridSize
  );

  device.queue.writeTexture(
    { texture: pressureBackTexture },
    pressureData,
    { bytesPerRow: gridSize.width * FLOAT_BYTES },
    gridSize
  );
}

export default SmokeSimulation;
