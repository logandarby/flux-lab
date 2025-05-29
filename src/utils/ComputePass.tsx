import type { TextureManager } from "./TextureManager";

export interface ComputePassConfig {
  readonly name: string;
  readonly shader: GPUShaderModule;
  readonly entryPoint: string;
}

export abstract class ComputePass<TextureID extends string | number> {
  protected readonly pipeline: GPUComputePipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(
    protected readonly config: ComputePassConfig,
    protected readonly device: GPUDevice
  ) {
    const { pipeline, bindGroupLayout } = this.setupPipeline();
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
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
  protected abstract createBindGroup(
    textureManager: TextureManager<TextureID>
  ): GPUBindGroup;

  public execute(
    pass: GPUComputePassEncoder,
    textureManager: TextureManager<TextureID>,
    workgroupCount: number
  ): void {
    const bindGroup = this.createBindGroup(textureManager);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);

    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
  }
}
