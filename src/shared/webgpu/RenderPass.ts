import type { TextureManager } from "./TextureManager";

export interface RenderPassConfig {
  name: string;
  vertex: GPUVertexState;
  fragment: GPUFragmentState;
}

export interface RenderPassBindArgs<TextureID extends string | number> {
  textureManager: TextureManager<TextureID>;
  sampler: GPUSampler;
  texture: TextureID;
}

interface NoiseArguments {
  stddev: number;
  mean: number;
  offsets: [number, number, number];
}

export interface RenderUniformWriteArgs {
  shaderMode: ShaderMode;
  noise: NoiseArguments;
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
  PARTICLES = 3,
}

export class RenderPass<TextureID extends string | number> {
  protected readonly pipeline: GPURenderPipeline;
  protected readonly bindGroupLayout: GPUBindGroupLayout;
  protected readonly uniformBuffer: GPUBuffer;

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
      size: 8 * 4, // 1 unsigned int + 5 floats
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
    texture,
  }: RenderPassBindArgs<TextureID>): GPUBindGroup {
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

  public writeToUniformBuffer({
    noise,
    shaderMode,
  }: RenderUniformWriteArgs): void {
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Uint32Array([shaderMode])
    );
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      4,
      new Float32Array([noise.stddev, noise.mean, ...noise.offsets])
    );
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

  public destroy(): void {
    this.uniformBuffer.destroy();
  }
}
