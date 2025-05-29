import type { TextureManager } from "@/SmokeSimulation";

export interface RenderPassConfig {
  name: string;
  vertex: GPUVertexState;
  fragment: GPUFragmentState;
}

export interface RenderPassDrawConfig {
  vertexCount: number;
  instanceCount?: number;
}

export abstract class RenderPass<TextureID extends string | number> {
  protected readonly pipeline: GPURenderPipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(
    protected readonly config: RenderPassConfig,
    protected readonly device: GPUDevice
  ) {
    this.bindGroupLayout = this.createBindGroupLayout();
    this.pipeline = this.device.createRenderPipeline({
      label: `${this.config.name} Pipeline`,
      layout: this.device.createPipelineLayout({
        label: `${this.config.name} Pipeline Layout`,
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: this.config.vertex,
      fragment: this.config.fragment,
    });
  }

  protected abstract createBindGroupLayout(): GPUBindGroupLayout;
  protected abstract createBindGroup(
    textureManager?: TextureManager<TextureID>
  ): GPUBindGroup;

  public execute(
    pass: GPURenderPassEncoder,
    drawConfig: RenderPassDrawConfig,
    textureManager?: TextureManager<TextureID>
  ): void {
    const bindGroup = this.createBindGroup(textureManager);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(drawConfig.vertexCount, drawConfig.instanceCount);
  }
}
