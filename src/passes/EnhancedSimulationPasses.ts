import { ComputePass, type BindGroupArgs } from "../utils/ComputePass";
import { injectShaderVariables } from "../utils/webgpu.utils";
import type { SmokeTextureID } from "../SmokeSimulation";
import type { TextureManager } from "../utils/TextureManager";
import {
  AdvectionUniforms,
  DiffusionUniforms,
  AddSmokeUniforms,
  AddVelocityUniforms,
} from "../utils/UniformManager";

import advectionShaderTemplate from "../shaders/advectionShader.wgsl?raw";
import jacobiIterationShaderTemplate from "../shaders/jacobiIteration.wgsl?raw";
import addSmokeShaderTemplate from "../shaders/addSmokeShader.wgsl?raw";
import addVelocityShaderTemplate from "../shaders/addVelocityShader.wgsl?raw";

// Enhanced binding layout manager (shared with legacy system)
enum BindGroupLayoutType {
  READ_READ_WRITE_UNIFORM = "read_read_write_uniform",
  READ_WRITE_UNIFORM = "read_write_uniform",
  READ_READ_WRITE_1 = "pressure",
  GRADIENT = "gradient",
  BOUNDARY = "boundary",
}

const BIND_GROUP_LAYOUT_DESCRIPTOR_RECORD: Record<
  BindGroupLayoutType,
  GPUBindGroupLayoutDescriptor
> = {
  [BindGroupLayoutType.READ_READ_WRITE_UNIFORM]: {
    label: "Read-Read-Write-Uniform Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "rg32float",
          access: "write-only",
          viewDimension: "2d",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  },
  [BindGroupLayoutType.READ_WRITE_UNIFORM]: {
    label: "Read-Write-Uniform Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "r32float",
          access: "write-only",
          viewDimension: "2d",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  },
  [BindGroupLayoutType.READ_READ_WRITE_1]: {
    label: "Pressure Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "r32float",
          access: "write-only",
          viewDimension: "2d",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  },
  [BindGroupLayoutType.GRADIENT]: {
    label: "Gradient Subtraction Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "rg32float",
          access: "write-only",
          viewDimension: "2d",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  },
  [BindGroupLayoutType.BOUNDARY]: {
    label: "Boundary Conditions Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "rg32float",
          access: "write-only",
          viewDimension: "2d",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  },
};

class BindLayoutManager {
  private static readonly layoutMap = new Map<string, GPUBindGroupLayout>();

  public static getBindGroupLayout(
    device: GPUDevice,
    layoutType: BindGroupLayoutType
  ) {
    const cacheKey = `${device.label || "device"}_${layoutType}`;
    if (this.layoutMap.has(cacheKey)) {
      return this.layoutMap.get(cacheKey)!;
    }
    const layoutDesc = BIND_GROUP_LAYOUT_DESCRIPTOR_RECORD[layoutType];
    const layout = device.createBindGroupLayout(layoutDesc);
    this.layoutMap.set(cacheKey, layout);
    return layout;
  }
}

/**
 * Enhanced Velocity Advection Pass with managed uniforms
 */
export class EnhancedVelocityAdvectionPass extends ComputePass<
  SmokeTextureID,
  AdvectionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Enhanced Velocity Advection",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Enhanced Velocity Advection Shader",
          code: injectShaderVariables(advectionShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
            OUT_FORMAT: "rg32float",
          }),
        }),
      },
      device,
      true // Enable uniform manager
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.READ_READ_WRITE_UNIFORM
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Enhanced Velocity Advection Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args
            .textureManager!.getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 1,
          resource: args
            .textureManager!.getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 2,
          resource: args
            .textureManager!.getBackTexture("velocity")
            .createView(),
        },
        {
          binding: 3,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }

  /**
   * Type-safe execution with automatic uniform management
   */
  public executeWithAdvectionParams(
    pass: GPUComputePassEncoder,
    textureManager: TextureManager<SmokeTextureID>,
    timestep: number,
    advectionFactor: number,
    workgroupCount: number
  ): void {
    const uniformData = new AdvectionUniforms(timestep, advectionFactor);

    this.executeWithUniforms(
      pass,
      {
        textureManager,
        uniformData,
      },
      workgroupCount
    );
  }
}

/**
 * Enhanced Diffusion Pass with managed uniforms and iteration support
 */
export class EnhancedDiffusionPass extends ComputePass<
  SmokeTextureID,
  DiffusionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Enhanced Diffusion",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Enhanced Diffusion Shader",
          code: injectShaderVariables(jacobiIterationShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
            FORMAT: "rg32float",
          }),
        }),
      },
      device,
      true // Enable uniform manager
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.READ_READ_WRITE_UNIFORM
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Enhanced Diffusion Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args
            .textureManager!.getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 1,
          resource: args
            .textureManager!.getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 2,
          resource: args
            .textureManager!.getBackTexture("velocity")
            .createView(),
        },
        {
          binding: 3,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }

  /**
   * Type-safe iteration execution with automatic uniform management
   */
  public executeIterationsWithParams(
    pass: GPUComputePassEncoder,
    textureManager: TextureManager<SmokeTextureID>,
    timestep: number,
    diffusionFactor: number,
    workgroupCount: number,
    iterations: number
  ): void {
    const uniformData = new DiffusionUniforms(timestep, diffusionFactor);

    for (let i = 0; i < iterations; i++) {
      this.executeWithUniforms(
        pass,
        {
          textureManager,
          uniformData,
        },
        workgroupCount
      );

      // Swap textures for next iteration
      textureManager?.swap("velocity");
    }
  }
}

/**
 * Enhanced Add Smoke Pass with managed uniforms
 */
export class EnhancedAddSmokePass extends ComputePass<
  SmokeTextureID,
  AddSmokeUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Enhanced Add Smoke",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Enhanced Add Smoke Shader",
          code: injectShaderVariables(addSmokeShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device,
      true // Enable uniform manager
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Enhanced Add Smoke Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "r32float",
            access: "write-only",
            viewDimension: "2d",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Enhanced Add Smoke Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args
            .textureManager!.getCurrentTexture("smokeDensity")
            .createView(),
        },
        {
          binding: 1,
          resource: args
            .textureManager!.getBackTexture("smokeDensity")
            .createView(),
        },
        {
          binding: 2,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }

  /**
   * Type-safe smoke addition with automatic uniform management
   */
  public addSmokeAtPosition(
    pass: GPUComputePassEncoder,
    textureManager: TextureManager<SmokeTextureID>,
    x: number,
    y: number,
    radius: number,
    intensity: number,
    workgroupCount: number
  ): void {
    const uniformData = new AddSmokeUniforms(x, y, radius, intensity);

    this.executeWithUniforms(
      pass,
      {
        textureManager,
        uniformData,
      },
      workgroupCount
    );
  }
}

/**
 * Enhanced Add Velocity Pass with managed uniforms
 */
export class EnhancedAddVelocityPass extends ComputePass<
  SmokeTextureID,
  AddVelocityUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Enhanced Add Velocity",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Enhanced Add Velocity Shader",
          code: injectShaderVariables(addVelocityShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device,
      true // Enable uniform manager
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Enhanced Add Velocity Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "rg32float",
            access: "write-only",
            viewDimension: "2d",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Enhanced Add Velocity Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args
            .textureManager!.getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 1,
          resource: args
            .textureManager!.getBackTexture("velocity")
            .createView(),
        },
        {
          binding: 2,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }

  /**
   * Type-safe velocity addition with automatic uniform management
   */
  public addVelocityAtPosition(
    pass: GPUComputePassEncoder,
    textureManager: TextureManager<SmokeTextureID>,
    x: number,
    y: number,
    velocityX: number,
    velocityY: number,
    radius: number,
    intensity: number,
    workgroupCount: number
  ): void {
    const uniformData = new AddVelocityUniforms(
      x,
      y,
      velocityX,
      velocityY,
      radius,
      intensity
    );

    this.executeWithUniforms(
      pass,
      {
        textureManager,
        uniformData,
      },
      workgroupCount
    );
  }
}
