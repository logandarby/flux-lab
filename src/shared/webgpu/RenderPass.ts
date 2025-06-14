import type { TextureManager } from "./TextureManager";
import { UniformBufferWriter } from "./UniformManager";

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
  smokeColor: [number, number, number];
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
      size: 12 * 4, // 1 unsigned int + 8 floats, padded to 16-byte alignment (48 bytes total)
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
    smokeColor,
  }: RenderUniformWriteArgs): void {
    const uniformData = new ArrayBuffer(48);
    const writer = new UniformBufferWriter(uniformData);

    writer.writeUint32(shaderMode);
    writer.writeFloat32(noise.stddev);
    writer.writeFloat32(noise.mean);
    writer.writeVec3f(noise.offsets[0], noise.offsets[1], noise.offsets[2]);
    writer.writeVec3f(smokeColor[0], smokeColor[1], smokeColor[2]);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
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
