import { useRef } from "react";
import BaseTestComponent from "./BaseTestComponent";
import { TextureManager } from "@/shared/webgpu/TextureManager";
import {
  type WebGPUResources,
  WebGPUError,
  WebGPUErrorCode,
  initializeWebGPU,
} from "@/shared/webgpu/webgpu.utils";
import type { SmokeTextureID } from "../../core/types";
import { ComputePass, type BindGroupArgs } from "@/shared/webgpu/ComputePass";
import { RenderPass, ShaderMode } from "@/shared/webgpu/RenderPass";
import { wgsl } from "@/lib/preprocessor";
import { SHADERS } from "../../shaders";

const GRID_SIZE = 16; // 16x16 grid
const WORKGROUP_SIZE = 8;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
const JACOBI_ITERATIONS = 20;

function initializeDiffusionField(
  diffusionTexture: GPUTexture,
  device: GPUDevice
): void {
  const _ = [0.0, 0.0]; // RG channels for 32-bit float (no diffusion)
  const d = [1.0, 1.0]; // RG channels for high diffusion value

  // prettier-ignore
  const diffusionData = new Float32Array([
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, d, d, d, _, _, _, _, _, _, _,
    _, _, _, _, _, _, d, d, d, _, _, _, _, _, _, _,
    _, _, _, _, _, _, d, d, d, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
  ].flat());

  device.queue.writeTexture(
    { texture: diffusionTexture },
    diffusionData,
    {
      bytesPerRow: GRID_SIZE * 2 * 4, // 2 channels × 4 bytes per 32-bit float
    },
    { width: GRID_SIZE, height: GRID_SIZE }
  );
}

class DiffusionPass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice) {
    super(
      {
        entryPoint: "compute_main",
        name: "Diffusion",
        shader: device.createShaderModule({
          code: wgsl(SHADERS.JACOBI_ITERATION, {
            variables: {
              WORKGROUP_SIZE,
              FORMAT: "rg32float",
            },
          }),
          label: `Diffusion Shader`,
        }),
      },
      device
    );
  }
  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: `${this.config.name} Bind Group Layout`,
      entries: [
        // x_in
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            viewDimension: "2d",
            sampleType: "unfilterable-float",
          },
        },
        // b_in
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            viewDimension: "2d",
            sampleType: "unfilterable-float",
          },
        },
        // texture_out
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "rg32float",
            viewDimension: "2d",
            access: "write-only",
          },
        },
        // uniforms
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
          },
        },
      ],
    });
  }

  protected createBindGroup({
    textureManager,
    uniformBuffer,
  }: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    if (!textureManager) {
      throw new Error("Supply texture manager to diffusion pass");
    }
    if (!uniformBuffer) {
      throw new Error("Supply uniform buffer to diffusion pass");
    }
    return this.device.createBindGroup({
      label: "Diffusion Bind Layout",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: textureManager.getCurrentTexture("velocity").createView(),
        },
        {
          binding: 1,
          resource: textureManager.getCurrentTexture("velocity").createView(),
        },
        {
          binding: 2,
          resource: textureManager.getBackTexture("velocity").createView(),
        },
        {
          binding: 3,
          resource: {
            label: "",
            buffer: uniformBuffer,
          },
        },
      ],
    });
  }

  public override async execute(
    pass: GPUComputePassEncoder,
    bindGroupArgs: BindGroupArgs<SmokeTextureID>,
    workgroupCount: number
  ) {
    for (let _ = 0; _ < JACOBI_ITERATIONS; _++) {
      const bindGroup = this.createBindGroup(bindGroupArgs);

      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);

      pass.dispatchWorkgroups(workgroupCount, workgroupCount);

      bindGroupArgs.textureManager?.swap("velocity");
    }
  }
}

class DiffusionTestSimulation {
  private resources: WebGPUResources | null = null;
  private textureManager: TextureManager<SmokeTextureID> | null = null;
  private sampler: GPUSampler | null = null;
  private diffusionPass: DiffusionPass | null = null; // Reusing AdvectionPass for now
  private renderingPass: RenderPass<SmokeTextureID> | null = null;

  public async initialize(canvasRef: React.RefObject<HTMLCanvasElement>) {
    if (!canvasRef.current) {
      throw new WebGPUError(
        "Could not initialize WebGPU: Canvas not found",
        WebGPUErrorCode.NO_CANVAS
      );
    }
    this.resources = await initializeWebGPU(canvasRef.current);

    // Initialize texture manager
    this.textureManager = new TextureManager<SmokeTextureID>(
      this.resources.device
    );

    this.sampler = this.resources.device.createSampler({
      label: "Diffusion Rendering Sampler ",
    });

    // Create diffusion texture (ping-pong for diffusion) - using "velocity" texture ID
    this.textureManager.createPingPongTexture("velocity", {
      label: "Velocity Texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Initialize diffusion field with test pattern
    initializeDiffusionField(
      this.textureManager.getCurrentTexture("velocity"),
      this.resources.device
    );

    // Create diffusion compute pass (reusing AdvectionPass for now)
    this.diffusionPass = new DiffusionPass(this.resources.device);

    // Create rendering pass
    const textureShaderModule = this.resources.device.createShaderModule({
      label: "Texture Shader",
      code: wgsl(SHADERS.TEXTURE),
    });
    this.renderingPass = new RenderPass<SmokeTextureID>(
      {
        name: "Diffusion Test Rendering",
        vertex: {
          module: textureShaderModule,
          entryPoint: "vertex_main",
        },
        fragment: {
          module: textureShaderModule,
          entryPoint: "fragment_main",
          targets: [
            {
              format: this.resources.canvasFormat,
            },
          ],
        },
      },
      this.resources.device
    );

    // Render the initial state
    this.step({ renderOnly: true });
  }

  public step({ renderOnly }: { renderOnly: boolean } = { renderOnly: false }) {
    if (
      !this.resources ||
      !this.textureManager ||
      !this.diffusionPass ||
      !this.renderingPass
    ) {
      throw new WebGPUError(
        "Could not step diffusion test: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    if (!renderOnly) {
      // Create uniform values for the diffusion pass
      const timestep = 1 / 30;
      const diffusionFactor = 0.01;
      const alpha = 1 / diffusionFactor / timestep;
      const uniformValues = new Float32Array([alpha, 1 / (4 + alpha)]);
      const uniformBuffer = this.resources.device.createBuffer({
        label: "Diffusion UBO",
        size: uniformValues.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
      });
      this.resources.device.queue.writeBuffer(uniformBuffer, 0, uniformValues);
      // Compute Pass - Run Diffusion (using advection pass for now)
      const diffusionPassEncoder = commandEncoder.beginComputePass({
        label: "Diffusion Test Compute Pass",
      });
      this.diffusionPass.execute(
        diffusionPassEncoder,
        {
          uniformBuffer,
          textureManager: this.textureManager,
        },
        WORKGROUP_COUNT
      );
      diffusionPassEncoder.end();

      // Swap textures for next iteration
      this.textureManager.swap("velocity");
    }

    // Render Pass - Display diffusion field
    const renderPassEncoder = commandEncoder.beginRenderPass({
      label: "Diffusion Test Render Pass",
      colorAttachments: [
        {
          view: this.resources.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        },
      ],
    });

    this.renderingPass.writeToUniformBuffer({
      shaderMode: ShaderMode.VELOCITY,
      noise: {
        mean: 0,
        stddev: 0.05,
        offsets: [Math.random(), Math.random(), Math.random()],
      },
    });
    this.renderingPass.execute(
      renderPassEncoder,
      {
        vertexCount: 6,
      },
      {
        sampler: this.sampler!,
        textureManager: this.textureManager,
        texture: "velocity",
      }
    );

    renderPassEncoder.end();
    this.resources.device.queue.submit([commandEncoder.finish()]);
  }
}

function DiffusionTestComponent() {
  const diffusionTestSimulation = useRef<DiffusionTestSimulation | null>(null);

  const handleInitialize = async (
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => {
    if (!canvasRef.current || diffusionTestSimulation.current) {
      return;
    }

    diffusionTestSimulation.current = new DiffusionTestSimulation();
    await diffusionTestSimulation.current.initialize(canvasRef);
  };

  const handleStep = () => {
    if (diffusionTestSimulation.current) {
      diffusionTestSimulation.current.step();
    }
  };

  const handleRestart = async (
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => {
    if (diffusionTestSimulation.current && canvasRef.current) {
      await diffusionTestSimulation.current.initialize(canvasRef);
    }
  };

  return (
    <BaseTestComponent
      title="Diffusion Test"
      onInitialize={handleInitialize}
      onStep={handleStep}
      onRestart={handleRestart}
    />
  );
}

export default DiffusionTestComponent;
