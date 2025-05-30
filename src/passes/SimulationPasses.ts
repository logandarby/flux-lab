import { ComputePass, type BindGroupArgs } from "../utils/ComputePass";
import { injectShaderVariables } from "../utils/webgpu.utils";
import type { SmokeTextureID } from "../SmokeSimulation";

import advectionShaderTemplate from "../shaders/advectionShader.wgsl?raw";
import jacobiIterationShaderTemplate from "../shaders/jacobiIteration.wgsl?raw";
// import pressureJacobiShader from "../shaders/pressureJacobi.wgsl?raw";
import divergenceShaderTemplate from "../shaders/divergenceShader.wgsl?raw";
import gradientSubtractionShaderTemplate from "../shaders/gradientSubtractionShader.wgsl?raw";

// Common bind group layout types for optimization
enum BindGroupLayoutType {
  READ_READ_WRITE_UNIFORM = "read_read_write_uniform",
  READ_WRITE_UNIFORM = "read_write_uniform",
  // Special case for pressure calculation - output must be r32float instead of r32float
  PRESSURE = "pressure",
  // Special case for gradient subtraction - pressure is r32float, velocity/output is rg32float
  GRADIENT = "gradient",
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
  [BindGroupLayoutType.PRESSURE]: {
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
};

class BindLayoutManager {
  private static readonly layoutMap = new Map<string, GPUBindGroupLayout>();

  public static getBindGroupLayout(
    device: GPUDevice,
    layoutType: BindGroupLayoutType
  ) {
    const cacheKey = `${device.label || device.adapterInfo?.device || "device"}_${layoutType}`;
    if (this.layoutMap.has(cacheKey)) {
      return this.layoutMap.get(cacheKey)!;
    }
    const layoutDesc: GPUBindGroupLayoutDescriptor | null =
      BIND_GROUP_LAYOUT_DESCRIPTOR_RECORD[layoutType];
    if (!layoutDesc) {
      throw new Error(`Unknown layout type: ${layoutType}`);
    }
    const layout = device.createBindGroupLayout(layoutDesc);
    this.layoutMap.set(cacheKey, layout);
    return layout;
  }
}

/**
 * Advects quantities using the velocity field via semi-Lagrangian method
 */
export class AdvectionPass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Advection",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Advection Shader",
          code: injectShaderVariables(advectionShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device
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

  private validateArgs(
    args: BindGroupArgs<SmokeTextureID>,
    required: (keyof BindGroupArgs<SmokeTextureID>)[]
  ): void {
    for (const key of required) {
      if (!args[key]) {
        throw new Error(`${key} is required for ${this.config.name}`);
      }
    }
  }
}

/**
 * Diffuses the velocity field using Jacobi iterations to solve the diffusion equation
 */
export class DiffusionPass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Diffusion",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Diffusion Shader",
          code: injectShaderVariables(jacobiIterationShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
            FORMAT: "rg32float",
          }),
        }),
      },
      device
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

  public executeIterations(
    pass: GPUComputePassEncoder,
    bindGroupArgs: BindGroupArgs<SmokeTextureID>,
    workgroupCount: number,
    iterations: number
  ): void {
    for (let i = 0; i < iterations; i++) {
      const bindGroup = this.createBindGroup(bindGroupArgs);

      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount, workgroupCount);

      // Swap textures for next iteration
      bindGroupArgs.textureManager?.swap("velocity");
    }
  }

  private validateArgs(
    args: BindGroupArgs<SmokeTextureID>,
    required: (keyof BindGroupArgs<SmokeTextureID>)[]
  ): void {
    for (const key of required) {
      if (!args[key]) {
        throw new Error(`${key} is required for ${this.config.name}`);
      }
    }
  }
}

/**
 * Computes the divergence of the velocity field
 */
export class DivergencePass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Divergence",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Divergence Shader",
          code: injectShaderVariables(divergenceShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device
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

  private validateArgs(
    args: BindGroupArgs<SmokeTextureID>,
    required: (keyof BindGroupArgs<SmokeTextureID>)[]
  ): void {
    for (const key of required) {
      if (!args[key]) {
        throw new Error(`${key} is required for ${this.config.name}`);
      }
    }
  }
}

/**
 * Solves for pressure using Jacobi iterations to enforce incompressibility
 */
export class PressurePass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Pressure",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Pressure Shader",
          code: injectShaderVariables(jacobiIterationShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
            FORMAT: "r32float",
          }),
        }),
      },
      device
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return BindLayoutManager.getBindGroupLayout(
      this.device,
      BindGroupLayoutType.PRESSURE
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

  public executeIterations(
    pass: GPUComputePassEncoder,
    bindGroupArgs: BindGroupArgs<SmokeTextureID>,
    workgroupCount: number,
    iterations: number
  ): void {
    for (let i = 0; i < iterations; i++) {
      const bindGroup = this.createBindGroup(bindGroupArgs);

      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount, workgroupCount);

      // Swap pressure textures for next iteration
      bindGroupArgs.textureManager?.swap("pressure");
    }
  }

  private validateArgs(
    args: BindGroupArgs<SmokeTextureID>,
    required: (keyof BindGroupArgs<SmokeTextureID>)[]
  ): void {
    for (const key of required) {
      if (!args[key]) {
        throw new Error(`${key} is required for ${this.config.name}`);
      }
    }
  }
}

/**
 * Subtracts pressure gradient from velocity to enforce incompressibility
 */
export class GradientSubtractionPass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Gradient Subtraction",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Gradient Subtraction Shader",
          code: injectShaderVariables(gradientSubtractionShaderTemplate, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    // Special case for gradient subtraction - pressure is r32float, velocity/output is rg32float
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

  private validateArgs(
    args: BindGroupArgs<SmokeTextureID>,
    required: (keyof BindGroupArgs<SmokeTextureID>)[]
  ): void {
    for (const key of required) {
      if (!args[key]) {
        throw new Error(`${key} is required for ${this.config.name}`);
      }
    }
  }
}

/*
 * Uniform buffer utilities for creating consistent buffer data
 */
export class UniformBufferUtils {
  public static createAdvectionUniforms(
    timestep: number,
    velocityAdvectionFactor: number
  ): Float32Array {
    return new Float32Array([timestep, velocityAdvectionFactor]);
  }

  public static createDiffusionUniforms(
    timestep: number,
    diffusionFactor: number
  ): Float32Array {
    const alpha = 1 / diffusionFactor / timestep;
    // const alpha = () / (viscosity * timestep);
    const rBeta = 1.0 / (4.0 + alpha);
    return new Float32Array([alpha, rBeta]);
  }

  public static createDivergenceUniforms(gridScale: number): Float32Array {
    const halfRdx = 0.5 / gridScale;
    return new Float32Array([halfRdx]);
  }

  public static createPressureUniforms(gridScale: number): Float32Array {
    const rdx = 1.0 / gridScale;
    const alpha = -(rdx * rdx); // For pressure Poisson equation
    const rBeta = 0.25;
    return new Float32Array([alpha, rBeta]);
  }

  public static createGradientSubtractionUniforms(
    gridScale: number
  ): Float32Array {
    const halfRdx = 0.5 / gridScale;
    return new Float32Array([halfRdx]);
  }
}
