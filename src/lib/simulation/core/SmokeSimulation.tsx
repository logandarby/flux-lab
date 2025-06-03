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
import { PerformanceTracker } from "@/lib/performance";
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
} from "./types";
import { SIMULATION_CONSTANTS } from "./constants";
import { wgsl } from "@/lib/preprocessor/core/wgsl";
import { SHADERS } from "../shaders";

function initializeTextures(
  textureManager: TextureManager<SmokeTextureID>,
  device: GPUDevice
): void {
  const gridSize = SIMULATION_CONSTANTS.grid.size;
  const particleSize = SIMULATION_CONSTANTS.particles.smokeDimensions;
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
    { bytesPerRow: gridSize.width * 2 * 4 },
    gridSize
  );

  const smokeDensityTexture = textureManager.getCurrentTexture("smokeDensity");
  device.queue.writeTexture(
    { texture: smokeDensityTexture },
    smokeDensityData,
    { bytesPerRow: gridSize.width * 4 },
    gridSize
  );

  const smokeParticlePositionsTexture = textureManager.getCurrentTexture(
    "smokeParticlePosition"
  );
  device.queue.writeTexture(
    { texture: smokeParticlePositionsTexture },
    particlePositionData,
    { bytesPerRow: particleSize.width * 4 * 2 },
    particleSize
  );

  // Initialize pressure textures
  const pressureTexture = textureManager.getCurrentTexture("pressure");
  const pressureBackTexture = textureManager.getBackTexture("pressure");

  device.queue.writeTexture(
    { texture: pressureTexture },
    pressureData,
    { bytesPerRow: gridSize.width * 4 },
    gridSize
  );

  device.queue.writeTexture(
    { texture: pressureBackTexture },
    pressureData,
    { bytesPerRow: gridSize.width * 4 },
    gridSize
  );
}

class SmokeSimulation {
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
      size: SIMULATION_CONSTANTS.grid.size,
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create divergence texture
    this.textureManager.createTexture("divergence", {
      label: "Divergence texture",
      size: SIMULATION_CONSTANTS.grid.size,
      format: "r32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create pressure texture (ping-pong)
    this.textureManager.createPingPongTexture("pressure", {
      label: "Pressure Texture",
      size: SIMULATION_CONSTANTS.grid.size,
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Smoke particle texture (positions)
    this.textureManager.createPingPongTexture("smokeParticlePosition", {
      label: "Smoke Position Texture",
      size: SIMULATION_CONSTANTS.particles.smokeDimensions,
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Smoke density texture
    this.textureManager.createPingPongTexture("smokeDensity", {
      label: "Smoke Density Texture",
      size: SIMULATION_CONSTANTS.grid.size,
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
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.smokeAdvectionPass = new SmokeAdvectionPasss(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.smokeDiffusionPass = new SmokeDiffusionPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.smokeDissipationPass = new SmokeDissipationPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.velocityDissipationPass = new VelocityDissipationPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.diffusionPass = new DiffusionPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.divergencePass = new DivergencePass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.pressurePass = new PressurePass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.gradientSubtractionPass = new GradientSubtractionPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.boundaryConditionsPass = new BoundaryConditionsPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.advectParticlesPass = new AdvectParticlesPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );

    // Initialize user interaction passes
    this.addSmokePass = new AddSmokePass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
    );
    this.addVelocityPass = new AddVelocityPass(
      this.resources.device,
      SIMULATION_CONSTANTS.compute.workgroupSize
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
      SIMULATION_CONSTANTS.physics.maxTimestep
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
        SIMULATION_CONSTANTS.physics.velocityAdvection
      );
      this.advectionPass!.executeWithUniforms(
        computePassEncoder,
        advectionUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (velocityControls.enableDiffusion) {
      const diffusionUniforms = new DiffusionUniforms(
        realTimeTimestep,
        SIMULATION_CONSTANTS.physics.diffusionFactor
      );
      this.diffusionPass!.executeIterations(
        computePassEncoder,
        diffusionUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount,
        SIMULATION_CONSTANTS.iterations.diffusion
      );
    }

    if (velocityControls.enableDivergence) {
      const divergenceUniforms = new DivergenceUniforms(
        SIMULATION_CONSTANTS.grid.scale
      );
      this.divergencePass!.executeWithUniforms(
        computePassEncoder,
        divergenceUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
    }

    if (velocityControls.enablePressureProjection) {
      const pressureUniforms = new PressureUniforms(
        SIMULATION_CONSTANTS.grid.scale
      );
      this.pressurePass!.executeIterations(
        computePassEncoder,
        pressureUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount,
        SIMULATION_CONSTANTS.iterations.pressure
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
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
      this.textureManager!.swap("pressure");
    }

    if (velocityControls.enableGradientSubtraction) {
      const gradientUniforms = new GradientSubtractionUniforms(
        SIMULATION_CONSTANTS.grid.scale
      );
      this.gradientSubtractionPass!.executeWithUniforms(
        computePassEncoder,
        gradientUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount
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
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (velocityControls.enableDissipation) {
      const velocityDissipationUniforms = new DissipationUniforms(
        SIMULATION_CONSTANTS.physics.velocityDissipationFactor
      );
      this.velocityDissipationPass!.executeWithUniforms(
        computePassEncoder,
        velocityDissipationUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
      this.textureManager!.swap("velocity");
    }

    if (smokeControls.enableAdvection) {
      const smokeAdvectionUniforms = new AdvectionUniforms(
        realTimeTimestep,
        SIMULATION_CONSTANTS.physics.smokeAdvection
      );
      this.smokeAdvectionPass!.executeWithUniforms(
        computePassEncoder,
        smokeAdvectionUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount
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
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
      this.textureManager!.swap("smokeDensity");
    }

    if (smokeControls.enableDiffusion) {
      const smokeDiffusionUniforms = new DiffusionUniforms(
        realTimeTimestep,
        SIMULATION_CONSTANTS.physics.smokeDiffusion
      );
      this.smokeDiffusionPass!.executeIterations(
        computePassEncoder,
        smokeDiffusionUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount,
        SIMULATION_CONSTANTS.iterations.diffusion
      );
      this.textureManager!.swap("smokeDensity");
    }

    if (smokeControls.enableDissipation) {
      const smokeDissipationUniforms = new DissipationUniforms(
        SIMULATION_CONSTANTS.physics.smokeDissipationFactor
      );
      this.smokeDissipationPass!.executeWithUniforms(
        computePassEncoder,
        smokeDissipationUniforms,
        this.textureManager!,
        SIMULATION_CONSTANTS.compute.workgroupCount
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
        SIMULATION_CONSTANTS.compute.workgroupCount
      );
      this.textureManager!.swap("smokeParticlePosition");
    }

    computePassEncoder.end();
  }

  public getPerformanceMetrics() {
    return this.performanceTracker.getMetrics();
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
    initializeTextures(this.textureManager, this.resources.device);

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
      SIMULATION_CONSTANTS.interaction.radius,
      SIMULATION_CONSTANTS.interaction.smokeIntensity
    );

    const addSmokePassEncoder = commandEncoder.beginComputePass({
      label: "Add Smoke Compute Pass",
    });
    this.addSmokePass.executeWithUniforms(
      addSmokePassEncoder,
      addSmokeUniforms,
      this.textureManager,
      SIMULATION_CONSTANTS.compute.workgroupCount
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
      SIMULATION_CONSTANTS.interaction.radius,
      SIMULATION_CONSTANTS.interaction.velocityIntensity
    );

    const addVelocityPassEncoder = commandEncoder.beginComputePass({
      label: "Add Velocity Compute Pass",
    });
    this.addVelocityPass.executeWithUniforms(
      addVelocityPassEncoder,
      addVelocityUniforms,
      this.textureManager,
      SIMULATION_CONSTANTS.compute.workgroupCount
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

export default SmokeSimulation;
