import type { TextureManager } from "./TextureManager";

export interface ComputePassConfig {
  readonly name: string;
  readonly shader: GPUShaderModule;
  readonly entryPoint: string;
}

export interface BindGroupArgs<TID extends string | number> {
  textureManager?: TextureManager<TID>;
  uniformBuffer?: GPUBuffer;
}

export abstract class ComputePass<TextureID extends string | number> {
  protected pipeline!: GPUComputePipeline;
  protected bindGroupLayout!: GPUBindGroupLayout;
  private initialized = false;

  constructor(
    protected readonly config: ComputePassConfig,
    protected readonly device: GPUDevice
  ) {
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

  // To be implemented by subclasses
  // Create a bindgroup layout for the desired shader
  protected abstract createBindGroupLayout(): GPUBindGroupLayout;
  // Create a bindgroup for the desired shader-- to be cached in the subclass
  protected abstract createBindGroup(
    bindGroupArgs: BindGroupArgs<TextureID>
  ): GPUBindGroup;

  // Utility function to validate required arguments for the class
  protected validateArgs(
    args: BindGroupArgs<TextureID>,
    required: (keyof BindGroupArgs<TextureID>)[]
  ): void {
    for (const key of required) {
      if (!args[key]) {
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
}
