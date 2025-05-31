import type { TextureManager } from "./TextureManager";
import { UniformManager, type UniformData } from "./UniformManager";

export interface ComputePassConfig {
  readonly name: string;
  readonly shader: GPUShaderModule;
  readonly entryPoint: string;
}

export interface BindGroupArgs<TID extends string | number> {
  textureManager?: TextureManager<TID>;
  uniformBuffer?: GPUBuffer; // Legacy support - will be deprecated
}

/**
 * Enhanced BindGroupArgs that supports the new uniform system
 */
export interface EnhancedBindGroupArgs<
  TID extends string | number,
  TUniform extends UniformData = UniformData,
> extends BindGroupArgs<TID> {
  uniformData?: TUniform;
}

export abstract class ComputePass<
  TextureID extends string | number,
  TUniform extends UniformData = UniformData,
> {
  protected pipeline!: GPUComputePipeline;
  protected bindGroupLayout!: GPUBindGroupLayout;
  protected uniformManager?: UniformManager<TUniform>;
  private initialized = false;

  constructor(
    protected readonly config: ComputePassConfig,
    protected readonly device: GPUDevice,
    useUniformManager = false
  ) {
    if (useUniformManager) {
      this.uniformManager = new UniformManager<TUniform>(
        device,
        this.config.name
      );
    }
    this.initializePipeline();
  }

  private initializePipeline(): void {
    // Delay pipeline setup to allow derived classes to initialize
    const { pipeline, bindGroupLayout } = this.setupPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.initialized = true;
  }

  private setupPipeline(): {
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  } {
    const bindGroupLayout = this.createBindGroupLayout();
    const pipeline = this.device.createComputePipeline({
      label: `${this.config.name} Pipeline`,
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: this.config.shader,
        entryPoint: this.config.entryPoint,
      },
    });
    return { pipeline, bindGroupLayout };
  }

  protected abstract createBindGroupLayout(): GPUBindGroupLayout;

  // Legacy method - still supported
  protected abstract createBindGroup(
    bindGroupArgs: BindGroupArgs<TextureID>
  ): GPUBindGroup;

  /**
   * Enhanced bind group creation that supports the new uniform system
   */
  protected createEnhancedBindGroup(
    bindGroupArgs: EnhancedBindGroupArgs<TextureID, TUniform>
  ): GPUBindGroup {
    // Fallback to legacy method if no uniform data is provided
    if (!bindGroupArgs.uniformData) {
      return this.createBindGroup(bindGroupArgs);
    }

    if (!this.uniformManager) {
      throw new Error(
        `${this.config.name} was not initialized with uniform manager support`
      );
    }

    // Get managed uniform buffer
    const uniformBuffer = this.uniformManager.getBuffer(
      bindGroupArgs.uniformData
    );

    // Create enhanced bind group args with the managed buffer
    const enhancedArgs: BindGroupArgs<TextureID> = {
      ...bindGroupArgs,
      uniformBuffer,
    };

    return this.createBindGroup(enhancedArgs);
  }

  protected validateArgs(
    args: BindGroupArgs<TextureID> | EnhancedBindGroupArgs<TextureID, TUniform>,
    required: (keyof BindGroupArgs<TextureID>)[]
  ): void {
    for (const key of required) {
      if (key === "uniformBuffer") {
        // For uniform buffer, check if we have either the buffer or uniform data
        const enhancedArgs = args as EnhancedBindGroupArgs<TextureID, TUniform>;
        if (!args[key] && !enhancedArgs.uniformData) {
          throw new Error(
            `${key} or uniformData is required for ${this.config.name}`
          );
        }
      } else if (!args[key]) {
        throw new Error(`${key} is required for ${this.config.name}`);
      }
    }
  }

  public execute(
    pass: GPUComputePassEncoder,
    bindGroupArgs: BindGroupArgs<TextureID>,
    workgroupCount: number
  ): void {
    if (!this.initialized) {
      throw new Error(
        `Compute Pass "${this.config.name} is not intialized yet"`
      );
    }
    const bindGroup = this.createBindGroup(bindGroupArgs);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
  }

  /**
   * Enhanced execute method that supports the new uniform system
   */
  public executeWithUniforms(
    pass: GPUComputePassEncoder,
    bindGroupArgs: EnhancedBindGroupArgs<TextureID, TUniform>,
    workgroupCount: number
  ): void {
    if (!this.initialized) {
      throw new Error(
        `Compute Pass "${this.config.name} is not initialized yet"`
      );
    }
    const bindGroup = this.createEnhancedBindGroup(bindGroupArgs);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
  }

  /**
   * Cleanup method to destroy managed resources
   */
  public destroy(): void {
    this.uniformManager?.destroy();
  }
}
