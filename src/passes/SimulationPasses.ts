import { ComputePass, type BindGroupArgs } from "../utils/ComputePass";
import { injectShaderVariables } from "../utils/webgpu.utils";
import type { SmokeTextureID } from "../SmokeSimulation";

import advectionShaderTemplate from "../shaders/advectionShader.wgsl?raw";
import jacobiIterationShader from "../shaders/jacobiIteration.wgsl?raw";
import pressureJacobiShader from "../shaders/pressureJacobi.wgsl?raw";
import divergenceShaderTemplate from "../shaders/divergenceShader.wgsl?raw";
import gradientSubtractionShaderTemplate from "../shaders/gradientSubtractionShader.wgsl?raw";

// Constants
const DIFFUSION_ITERATIONS = 20;
const PRESSURE_ITERATIONS = 40;

// Common bind group layout types for optimization
enum BindGroupLayoutType {
  READ_READ_WRITE_UNIFORM = "read_read_write_uniform",
  READ_WRITE_UNIFORM = "read_write_uniform",
}

/**
 * Base class for simulation passes that provides common bind group layouts
 */
abstract class SimulationPass extends ComputePass<SmokeTextureID> {
  protected static bindGroupLayouts = new Map<string, GPUBindGroupLayout>();

  constructor(
    config: { name: string; entryPoint: string; shader: GPUShaderModule },
    device: GPUDevice,
    private readonly layoutType: BindGroupLayoutType
  ) {
    super(config, device);
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    if (!this.layoutType) {
      throw new Error("Layout type not initialized");
    }

    const cacheKey = `${this.device.label || "device"}_${this.layoutType}`;

    if (SimulationPass.bindGroupLayouts.has(cacheKey)) {
      return SimulationPass.bindGroupLayouts.get(cacheKey)!;
    }

    let layout: GPUBindGroupLayout;

    switch (this.layoutType) {
      case BindGroupLayoutType.READ_READ_WRITE_UNIFORM:
        layout = this.device.createBindGroupLayout({
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
        });
        break;

      case BindGroupLayoutType.READ_WRITE_UNIFORM:
        layout = this.device.createBindGroupLayout({
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
        });
        break;

      default:
        throw new Error(`Unknown layout type: ${this.layoutType}`);
    }

    SimulationPass.bindGroupLayouts.set(cacheKey, layout);
    return layout;
  }
}

/**
 * Advects quantities using the velocity field via semi-Lagrangian method
 */
export class AdvectionPass extends SimulationPass {
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
      device,
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
export class DiffusionPass extends SimulationPass {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Diffusion",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Diffusion Shader",
          code: injectShaderVariables(jacobiIterationShader, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device,
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

  public override async execute(
    pass: GPUComputePassEncoder,
    bindGroupArgs: BindGroupArgs<SmokeTextureID>,
    workgroupCount: number
  ): Promise<void> {
    for (let i = 0; i < DIFFUSION_ITERATIONS; i++) {
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
export class DivergencePass extends SimulationPass {
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
      device,
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
export class PressurePass extends SimulationPass {
  constructor(device: GPUDevice, workgroupSize: number) {
    super(
      {
        name: "Pressure",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Pressure Shader",
          code: injectShaderVariables(pressureJacobiShader, {
            WORKGROUP_SIZE: workgroupSize,
          }),
        }),
      },
      device,
      BindGroupLayoutType.READ_READ_WRITE_UNIFORM
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    // Special case for pressure - needs r32float output instead of rg32float
    const cacheKey = `${this.device.label || "device"}_pressure_layout`;

    if (SimulationPass.bindGroupLayouts.has(cacheKey)) {
      return SimulationPass.bindGroupLayouts.get(cacheKey)!;
    }

    const layout = this.device.createBindGroupLayout({
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
    });

    SimulationPass.bindGroupLayouts.set(cacheKey, layout);
    return layout;
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

  public override async execute(
    pass: GPUComputePassEncoder,
    bindGroupArgs: BindGroupArgs<SmokeTextureID>,
    workgroupCount: number
  ): Promise<void> {
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
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
export class GradientSubtractionPass extends SimulationPass {
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
      device,
      BindGroupLayoutType.READ_READ_WRITE_UNIFORM
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    // Special case for gradient subtraction - pressure is r32float, velocity/output is rg32float
    const cacheKey = `${this.device.label || "device"}_gradient_layout`;

    if (SimulationPass.bindGroupLayouts.has(cacheKey)) {
      return SimulationPass.bindGroupLayouts.get(cacheKey)!;
    }

    const layout = this.device.createBindGroupLayout({
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
    });

    SimulationPass.bindGroupLayouts.set(cacheKey, layout);
    return layout;
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

/**
 * Factory class for creating simulation passes
 */
export class SimulationPassFactory {
  public static createAdvectionPass(
    device: GPUDevice,
    workgroupSize: number
  ): AdvectionPass {
    return new AdvectionPass(device, workgroupSize);
  }

  public static createDiffusionPass(
    device: GPUDevice,
    workgroupSize: number
  ): DiffusionPass {
    return new DiffusionPass(device, workgroupSize);
  }

  public static createDivergencePass(
    device: GPUDevice,
    workgroupSize: number
  ): DivergencePass {
    return new DivergencePass(device, workgroupSize);
  }

  public static createPressurePass(
    device: GPUDevice,
    workgroupSize: number
  ): PressurePass {
    return new PressurePass(device, workgroupSize);
  }

  public static createGradientSubtractionPass(
    device: GPUDevice,
    workgroupSize: number
  ): GradientSubtractionPass {
    return new GradientSubtractionPass(device, workgroupSize);
  }
}

/**
 * Uniform buffer utilities for creating consistent buffer data
 */
export class UniformBufferUtils {
  public static createAdvectionUniforms(
    timestep: number,
    gridScale: number
  ): Float32Array {
    return new Float32Array([timestep, 1.0 / gridScale]);
  }

  public static createDiffusionUniforms(
    timestep: number,
    gridScale: number,
    viscosity: number
  ): Float32Array {
    const rdx = 1.0 / gridScale;
    const alpha = (rdx * rdx) / (viscosity * timestep);
    const rBeta = 1.0 / (4.0 + alpha);
    return new Float32Array([alpha, rBeta]);
  }

  public static createDivergenceUniforms(gridScale: number): Float32Array {
    const halfRdx = 0.5 / gridScale;
    return new Float32Array([halfRdx]);
  }

  public static createPressureUniforms(): Float32Array {
    const alpha = -1.0; // For pressure Poisson equation
    const rBeta = 0.25; // 1/(4 + alpha) = 1/(4 + (-1)) = 1/3, but we use 0.25 for stability
    return new Float32Array([alpha, rBeta]);
  }

  public static createGradientSubtractionUniforms(
    gridScale: number
  ): Float32Array {
    const halfRdx = 0.5 / gridScale;
    return new Float32Array([halfRdx]);
  }
}
