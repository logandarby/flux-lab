import type { TextureManager } from "./TextureManager";

export interface RenderPassConfig {
  name: string;
  vertex: GPUVertexState;
  fragment: GPUFragmentState;
}

export interface RenderPassBindArgs<TextureID extends string | number> {
  textureManager: TextureManager<TextureID>;
  sampler: GPUSampler;
  shaderMode: ShaderMode;
  texture: TextureID;
}

export interface RenderPassDrawConfig {
  vertexCount: number;
  instanceCount?: number;
}

// NOTE Must be kept in sync with texture shader
export enum ShaderMode {
  VELOCITY = 0,
  PRESSURE = 1,
  DENSITY = 2,
}

export class RenderPass<TextureID extends string | number> {
  protected readonly pipeline: GPURenderPipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;
  protected readonly uniformBuffer: GPUBuffer;
  protected readonly currentShaderMode: ShaderMode | null = null;

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
    this.uniformBuffer = this.device.createBuffer({
      label: `${this.config.name} Uniform Buffer`,
      size: 4, // 1 float
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform",
          },
        },
      ],
    });
  }
  protected createBindGroup({
    textureManager,
    sampler,
    shaderMode,
    texture,
  }: RenderPassBindArgs<TextureID>): GPUBindGroup {
    if (shaderMode !== this.currentShaderMode) {
      this.device.queue.writeBuffer(
        this.uniformBuffer,
        0,
        new Uint32Array([shaderMode])
      );
    }
    return this.device.createBindGroup({
      label: "Smoke Texture Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture(texture).createView(),
        },
        {
          binding: 1,
          resource: sampler,
        },
        {
          binding: 2,
          resource: {
            buffer: this.uniformBuffer,
          },
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
