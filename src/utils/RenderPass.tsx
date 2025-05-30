import type { TextureManager } from "./TextureManager";

export interface RenderPassConfig<TextureID extends string | number> {
  name: string;
  vertex: GPUVertexState;
  fragment: GPUFragmentState;
  outputTextureName: TextureID;
}

export interface RenderPassBindArgs<TextureID extends string | number> {
  textureManager: TextureManager<TextureID>;
  sampler: GPUSampler;
}

export interface RenderPassDrawConfig {
  vertexCount: number;
  instanceCount?: number;
}

export abstract class RenderPass<TextureID extends string | number> {
  protected readonly pipeline: GPURenderPipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;

  constructor(
    protected readonly config: RenderPassConfig<TextureID>,
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

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Texture Rendering Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
            multisampled: false,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {
            type: "non-filtering",
          },
        },
      ],
    });
  }
  protected createBindGroup({
    textureManager,
    sampler,
  }: RenderPassBindArgs<TextureID>): GPUBindGroup {
    return this.device.createBindGroup({
      label: "Smoke Texture Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager
            .getCurrentTexture(this.config.outputTextureName)
            .createView(),
        },
        {
          binding: 1,
          resource: sampler,
        },
      ],
    });
  }

  public execute(
    pass: GPURenderPassEncoder,
    drawConfig: RenderPassDrawConfig,
    args: RenderPassBindArgs<TextureID>
  ): void {
    const bindGroup = this.createBindGroup(args);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(drawConfig.vertexCount, drawConfig.instanceCount);
  }
}
