import { ComputePass, type BindGroupArgs } from "@/shared/webgpu/ComputePass";
import type { TextureManager } from "@/shared/webgpu/TextureManager";
import {
  UniformManager,
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
  type UniformData,
} from "@/shared/webgpu/UniformManager";

import type { SmokeTextureID } from "./types";
import { SHADERS, wgsl } from "../shaders";

/**
 * Boundary condition types for fluid simulation.
 * These control how velocities and scalar fields behave at the edges of the simulation grid.
 */
export enum BoundaryType {
  /** Velocity sticks to boundaries (fluid cannot flow through walls) */
  NO_SLIP_VELOCITY = 0,
  /** Velocity can slip along boundaries (frictionless walls) */
  FREE_SLIP_VELOCITY = 1,
  /** Scalar fields have zero gradient at boundaries (no flux through walls) */
  SCALAR_NEUMANN = 2,
}

/**
 * Optimized bind group layout types.
 * These layouts are cached and reused across passes to avoid recreation overhead.
 */
enum BindGroupLayoutType {
  /** Two input textures, one output texture, one uniform buffer */
  READ_READ_WRITE_UNIFORM = "read_read_write_uniform",
  /** One input texture, one output texture, one uniform buffer */
  READ_WRITE_UNIFORM = "read_write_uniform",
  /** Pressure-specific layout with r32float output format */
  READ_READ_WRITE_1 = "pressure",
  /** Gradient subtraction with mixed texture formats (pressure + velocity) */
  GRADIENT = "gradient",
  /** Boundary conditions with format-agnostic layouts */
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
        texture: {
          sampleType: "unfilterable-float",
          viewDimension: "2d",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: "unfilterable-float",
          viewDimension: "2d",
        },
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
        texture: {
          sampleType: "unfilterable-float",
          viewDimension: "2d",
        },
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
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" }, // pressure (r32float)
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" }, // velocity (rg32float)
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "rg32float",
          access: "write-only",
          viewDimension: "2d",
        }, // output velocity
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
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" }, // input field
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "rg32float", // Default format, will be overridden
          access: "write-only",
          viewDimension: "2d",
        }, // output field
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  },
};

/**
 * Manages cached bind group layouts to avoid recreation overhead.
 */
export class BindLayoutManager {
  private static readonly layoutMap = new Map<
    GPUDevice,
    Map<BindGroupLayoutType, GPUBindGroupLayout>
  >();

  public static getBindGroupLayout(
    device: GPUDevice,
    layoutType: BindGroupLayoutType
  ): GPUBindGroupLayout {
    // Get or create device-specific cache
    let deviceCache = this.layoutMap.get(device);
    if (!deviceCache) {
      deviceCache = new Map<BindGroupLayoutType, GPUBindGroupLayout>();
      this.layoutMap.set(device, deviceCache);
    }

    // Check if layout already exists for this device
    if (deviceCache.has(layoutType)) {
      return deviceCache.get(layoutType)!;
    }

    // Create new layout
    const layoutDesc: GPUBindGroupLayoutDescriptor | null =
      BIND_GROUP_LAYOUT_DESCRIPTOR_RECORD[layoutType];
    if (!layoutDesc) {
      throw new Error(`Unknown layout type: ${layoutType}`);
    }

    const layout = device.createBindGroupLayout(layoutDesc);
    deviceCache.set(layoutType, layout);
    return layout;
  }

  /**
   * Clean up cached layouts for a specific device.
   */
  public static destroyDevice(device: GPUDevice): void {
    const deviceCache = this.layoutMap.get(device);
    if (deviceCache) {
      // Note: WebGPU bind group layouts don't have a destroy() method
      // They are automatically cleaned up when the device is lost
      deviceCache.clear();
      this.layoutMap.delete(device);
    }
  }

  /**
   * Clean up all cached bind group layouts.
   */
  public static destroyAll(): void {
    for (const [_device, deviceCache] of this.layoutMap.entries()) {
      deviceCache.clear();
    }
    this.layoutMap.clear();
  }
}

/**
 * Base class for compute passes with uniform buffer management.
 */
abstract class UniformComputePass<
  TTextureID extends string,
  TUniform extends UniformData,
> extends ComputePass<TTextureID> {
  protected uniformManager: UniformManager<TUniform>;

  constructor(
    config: { name: string; entryPoint: string; shader: GPUShaderModule },
    device: GPUDevice,
    uniformLabel: string
  ) {
    super(config, device);
    this.uniformManager = new UniformManager<TUniform>(device, uniformLabel);
  }

  /**
   * Execute the compute pass with typed uniform data.
   */
  public executeWithUniforms(
    pass: GPUComputePassEncoder,
    uniformData: TUniform,
    textureManager: TextureManager<TTextureID>,
    workgroupCount: number
  ): void {
    const uniformBuffer = this.uniformManager.updateBuffer(uniformData);
    const bindGroup = this.createBindGroup({
      textureManager,
      uniformBuffer,
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
  }

  public destroy(): void {
    this.uniformManager.destroy();
  }
}

/**
 * Base class for iterative algorithms like diffusion and pressure solving.
 */
abstract class IterativeComputePass<
  TTextureID extends string,
  TUniform extends UniformData,
> extends UniformComputePass<TTextureID, TUniform> {
  private cachedBindGroupSwaps: [GPUBindGroup, GPUBindGroup] | null = null;

  /**
   * Subclasses must specify which texture to swap between iterations.
   */
  protected abstract getSwapTextureId(): TTextureID;

  protected getCachedBindGroups(
    uniformData: TUniform,
    textureManager: TextureManager<TTextureID>
  ): [GPUBindGroup, GPUBindGroup] {
    if (this.cachedBindGroupSwaps) {
      return this.cachedBindGroupSwaps;
    }
    const uniformBuffer = this.uniformManager.updateBuffer(uniformData);
    const bindGroup1 = this.createBindGroup({
      textureManager,
      uniformBuffer,
    });
    textureManager.swap(this.getSwapTextureId());
    const bindGroup2 = this.createBindGroup({
      textureManager,
      uniformBuffer,
    });
    textureManager.swap(this.getSwapTextureId());
    this.cachedBindGroupSwaps = [bindGroup1, bindGroup2];
    return this.cachedBindGroupSwaps;
  }

  /**
   * Execute multiple iterations of the pass.
   */
  public executeIterations(
    pass: GPUComputePassEncoder,
    uniformData: TUniform,
    textureManager: TextureManager<TTextureID>,
    workgroupCount: number,
    iterations: number
  ): void {
    const bindGroups = this.getCachedBindGroups(uniformData, textureManager);
    pass.setPipeline(this.pipeline);
    for (let i = 0; i < iterations; i++) {
      pass.setBindGroup(0, bindGroups[i % 2]);
      pass.dispatchWorkgroups(workgroupCount, workgroupCount);
    }
  }
}

/**
 * Advects smoke density using the velocity field.
 */
export class SmokeAdvectionPasss extends UniformComputePass<
  SmokeTextureID,
  AdvectionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Smoke Density Advection",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Advection Shader",
          code: wgsl(SHADERS.ADVECTION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              OUT_FORMAT: "r32float",
            },
          }),
        }),
      },
      device,
      "Smoke Advection Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.READ_READ_WRITE_1
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Advection Bind Group",
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
            .textureManager!.getCurrentTexture("smokeDensity")
            .createView(),
        },
        {
          binding: 2,
          resource: args
            .textureManager!.getBackTexture("smokeDensity")
            .createView(),
        },
        {
          binding: 3,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }
}

/**
 * Diffuses smoke density using Jacobi iteration.
 */
export class SmokeDiffusionPass extends IterativeComputePass<
  SmokeTextureID,
  DiffusionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Diffusion",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Diffusion Shader",
          code: wgsl(SHADERS.JACOBI_ITERATION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              FORMAT: "r32float",
            },
          }),
        }),
      },
      device,
      "Smoke Diffusion Uniforms"
    );
  }

  protected getSwapTextureId(): SmokeTextureID {
    return "smokeDensity";
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.READ_READ_WRITE_1
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Diffusion Bind Group",
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
            .textureManager!.getCurrentTexture("smokeDensity")
            .createView(),
        },
        {
          binding: 2,
          resource: args
            .textureManager!.getBackTexture("smokeDensity")
            .createView(),
        },
        {
          binding: 3,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }
}

/**
 * Advects velocity using the velocity field itself.
 */
export class VelocityAdvectionPass extends UniformComputePass<
  SmokeTextureID,
  AdvectionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Advection",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Advection Shader",
          code: wgsl(SHADERS.ADVECTION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              OUT_FORMAT: "rg32float",
            },
          }),
        }),
      },
      device,
      "Velocity Advection Uniforms"
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
      label: "Advection Bind Group",
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
}

/**
 * Diffuses velocity using Jacobi iteration.
 */
export class DiffusionPass extends IterativeComputePass<
  SmokeTextureID,
  DiffusionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Diffusion",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Diffusion Shader",
          code: wgsl(SHADERS.JACOBI_ITERATION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              FORMAT: "rg32float",
            },
          }),
        }),
      },
      device,
      "Velocity Diffusion Uniforms"
    );
  }

  protected getSwapTextureId(): SmokeTextureID {
    return "velocity";
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
      label: "Diffusion Bind Group",
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
}

/**
 * Computes the divergence of the velocity field.
 */
export class DivergencePass extends UniformComputePass<
  SmokeTextureID,
  DivergenceUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Divergence",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Divergence Shader",
          code: wgsl(SHADERS.DIVERGENCE, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
            },
          }),
        }),
      },
      device,
      "Divergence Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.READ_WRITE_UNIFORM
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Divergence Bind Group",
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
            .textureManager!.getCurrentTexture("divergence")
            .createView(),
        },
        {
          binding: 2,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }
}

/**
 * Solves for pressure using Jacobi iteration.
 */
export class PressurePass extends IterativeComputePass<
  SmokeTextureID,
  PressureUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Pressure",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Pressure Shader",
          code: wgsl(SHADERS.JACOBI_ITERATION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              FORMAT: "r32float",
            },
          }),
        }),
      },
      device,
      "Pressure Uniforms"
    );
  }

  protected getSwapTextureId(): SmokeTextureID {
    return "pressure";
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.READ_READ_WRITE_1
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Pressure Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args
            .textureManager!.getCurrentTexture("pressure")
            .createView(),
        },
        {
          binding: 1,
          resource: args
            .textureManager!.getCurrentTexture("divergence")
            .createView(),
        },
        {
          binding: 2,
          resource: args
            .textureManager!.getBackTexture("pressure")
            .createView(),
        },
        {
          binding: 3,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }
}

/**
 * Subtracts pressure gradient from velocity to enforce incompressibility.
 */
export class GradientSubtractionPass extends UniformComputePass<
  SmokeTextureID,
  GradientSubtractionUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Gradient Subtraction",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Gradient Subtraction Shader",
          code: wgsl(SHADERS.GRADIENT_SUBTRACTION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
            },
          }),
        }),
      },
      device,
      "Gradient Subtraction Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.GRADIENT
    );
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Gradient Subtraction Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args
            .textureManager!.getCurrentTexture("pressure")
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
}

/**
 * Enforces boundary conditions on velocity and scalar fields.
 */
export class BoundaryConditionsPass extends UniformComputePass<
  SmokeTextureID,
  BoundaryUniforms
> {
  private velocityShader: GPUShaderModule;
  private scalarShader: GPUShaderModule;
  private velocityPipeline: GPUComputePipeline | null = null;
  private scalarPipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice, workgroupSize: number) {
    // Create specialized shaders for different texture formats
    const velocityShaderCode = wgsl(SHADERS.BOUNDARY_CONDITIONS, {
      variables: {
        WORKGROUP_SIZE: workgroupSize,
        FORMAT: "rg32float",
      },
    });

    const scalarShaderCode = wgsl(SHADERS.BOUNDARY_CONDITIONS, {
      variables: {
        WORKGROUP_SIZE: workgroupSize,
        FORMAT: "r32float",
      },
    });

    const velocityShader = device.createShaderModule({
      label: "Boundary Conditions Velocity Shader",
      code: velocityShaderCode,
    });

    const scalarShader = device.createShaderModule({
      label: "Boundary Conditions Scalar Shader",
      code: scalarShaderCode,
    });

    super(
      {
        name: "Boundary Conditions",
        entryPoint: "compute_main",
        shader: velocityShader,
      },
      device,
      "Boundary Conditions Uniforms"
    );

    this.velocityShader = velocityShader;
    this.scalarShader = scalarShader;
  }

  private getVelocityBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Boundary Velocity Bind Group Layout",
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

  private getScalarBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Boundary Scalar Bind Group Layout",
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

  private getVelocityPipeline(): GPUComputePipeline {
    if (!this.velocityPipeline) {
      this.velocityPipeline = this.device.createComputePipeline({
        label: "Boundary Conditions Velocity Pipeline",
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [this.getVelocityBindGroupLayout()],
        }),
        compute: {
          module: this.velocityShader,
          entryPoint: "compute_main",
        },
      });
    }
    return this.velocityPipeline;
  }

  private getScalarPipeline(): GPUComputePipeline {
    if (!this.scalarPipeline) {
      this.scalarPipeline = this.device.createComputePipeline({
        label: "Boundary Conditions Scalar Pipeline",
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [this.getScalarBindGroupLayout()],
        }),
        compute: {
          module: this.scalarShader,
          entryPoint: "compute_main",
        },
      });
    }
    return this.scalarPipeline;
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.getVelocityBindGroupLayout();
  }

  protected createBindGroup(_: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    throw new Error("Use executeForTexture method instead");
  }

  /**
   * Execute boundary conditions for a specific texture type.
   */
  public executeForTexture(
    pass: GPUComputePassEncoder,
    textureId: SmokeTextureID,
    uniformData: BoundaryUniforms,
    textureManager: TextureManager<SmokeTextureID>,
    workgroupCount: number
  ): void {
    const isVelocityTexture = textureId === "velocity";
    const pipeline = isVelocityTexture
      ? this.getVelocityPipeline()
      : this.getScalarPipeline();

    const uniformBuffer = this.uniformManager.updateBuffer(uniformData);

    const bindGroup = this.device.createBindGroup({
      label: `Boundary ${textureId} Bind Group`,
      layout: isVelocityTexture
        ? this.getVelocityBindGroupLayout()
        : this.getScalarBindGroupLayout(),
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture(textureId).createView(),
        },
        {
          binding: 1,
          resource: textureManager.getBackTexture(textureId).createView(),
        },
        {
          binding: 2,
          resource: { buffer: uniformBuffer },
        },
      ],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
  }

  public override destroy(): void {
    // Clean up additional pipelines
    this.velocityPipeline = null;
    this.scalarPipeline = null;

    // Call parent destroy to clean up uniform manager
    super.destroy();
  }
}

/**
 * Adds smoke density at a specific position.
 */
export class AddSmokePass extends UniformComputePass<
  SmokeTextureID,
  AddSmokeUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Add Smoke",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Add Smoke Shader",
          code: wgsl(SHADERS.ADD_SMOKE, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
            },
          }),
        }),
      },
      device,
      "Add Smoke Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Add Smoke Bind Group Layout",
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
      label: "Add Smoke Bind Group",
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
}

/**
 * Adds velocity at a specific position based on mouse movement.
 */
export class AddVelocityPass extends UniformComputePass<
  SmokeTextureID,
  AddVelocityUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Add Velocity",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Add Velocity Shader",
          code: wgsl(SHADERS.ADD_VELOCITY, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
            },
          }),
        }),
      },
      device,
      "Add Velocity Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Add Velocity Bind Group Layout",
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
      label: "Add Velocity Bind Group",
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
}

/**
 * Applies dissipation to smoke density, gradually reducing it over time.
 *
 * This simulates how smoke naturally fades away due to mixing with air,
 * cooling, or other physical processes. Without dissipation, smoke would
 * persist forever in the simulation.
 */
export class SmokeDissipationPass extends UniformComputePass<
  SmokeTextureID,
  DissipationUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Smoke Dissipation",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Smoke Dissipation Shader",
          code: wgsl(SHADERS.DISSIPATION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              FORMAT: "r32float",
              CHANNELS: 1,
            },
          }),
        }),
      },
      device,
      "Smoke Dissipation Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Smoke Dissipation Bind Group Layout",
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
      label: "Smoke Dissipation Bind Group",
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
}

export class AdvectParticlesPass extends UniformComputePass<
  SmokeTextureID,
  AdvectParticlesUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Advect Particles",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Advect Particles Shader",
          code: wgsl(SHADERS.ADVECT_PARTICLES, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
            },
          }),
        }),
      },
      device,
      "Advect Particles Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Advect Particles Bind Group Layout",
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
    });
  }

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    this.validateArgs(args, ["textureManager", "uniformBuffer"]);

    return this.device.createBindGroup({
      label: "Advect Particles Bind Group",
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
            .textureManager!.getCurrentTexture("smokeParticlePosition")
            .createView(),
        },
        {
          binding: 2,
          resource: args
            .textureManager!.getBackTexture("smokeParticlePosition")
            .createView(),
        },
        {
          binding: 3,
          resource: { buffer: args.uniformBuffer! },
        },
      ],
    });
  }
}

/**
 * Applies dissipation to velocity field, gradually reducing it over time.
 *
 * This simulates friction and other energy loss mechanisms that cause
 * fluid motion to slow down over time. Without velocity dissipation,
 * the fluid would maintain motion indefinitely.
 */
export class VelocityDissipationPass extends UniformComputePass<
  SmokeTextureID,
  DissipationUniforms
> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Velocity Dissipation",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Velocity Dissipation Shader",
          code: wgsl(SHADERS.DISSIPATION, {
            variables: {
              WORKGROUP_SIZE: workgroupSize,
              FORMAT: "rg32float",
              CHANNELS: 2,
            },
          }),
        }),
      },
      device,
      "Velocity Dissipation Uniforms"
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Velocity Dissipation Bind Group Layout",
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
      label: "Velocity Dissipation Bind Group",
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
}
