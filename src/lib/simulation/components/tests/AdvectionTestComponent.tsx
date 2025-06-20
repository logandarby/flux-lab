import { useRef, useState } from "react";
import BaseTestComponent from "./BaseTestComponent";
import { TextureManager } from "@/shared/webgpu/TextureManager";
import {
  type WebGPUResources,
  WebGPUError,
  WebGPUErrorCode,
  initializeWebGPU,
} from "@/shared/webgpu/webgpu.utils";
import { type SmokeTextureID } from "../../core/types";
import { ComputePass, type BindGroupArgs } from "@/shared/webgpu/ComputePass";
import { RenderPass, ShaderMode } from "@/shared/webgpu/RenderPass";
import { wgsl } from "@/lib/preprocessor";
import { SHADERS } from "../../shaders";

const GRID_SIZE = 16; // 8x8 grid
const WORKGROUP_SIZE = 8;
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
const ADVECTION_FACTOR = 0.5;

// Direction enumeration for cardinal and ordinal directions
enum AdvectionDirection {
  NORTH = "N",
  NORTHEAST = "NE",
  EAST = "E",
  SOUTHEAST = "SE",
  SOUTH = "S",
  SOUTHWEST = "SW",
  WEST = "W",
  NORTHWEST = "NW",
}

// Direction to velocity vector mapping
const DIRECTION_VECTORS: Record<AdvectionDirection, [number, number]> = {
  [AdvectionDirection.NORTH]: [0, -1],
  [AdvectionDirection.NORTHEAST]: [1, -1],
  [AdvectionDirection.EAST]: [1, 0],
  [AdvectionDirection.SOUTHEAST]: [1, 1],
  [AdvectionDirection.SOUTH]: [0, 1],
  [AdvectionDirection.SOUTHWEST]: [-1, 1],
  [AdvectionDirection.WEST]: [-1, 0],
  [AdvectionDirection.NORTHWEST]: [-1, -1],
};

function initializeVelocityField(
  velocityTexture: GPUTexture,
  device: GPUDevice,
  direction: AdvectionDirection = AdvectionDirection.EAST
): void {
  const _ = [0.0, 0.0]; // RG channels for 32-bit float (no velocity)
  const velocityVector = DIRECTION_VECTORS[direction];
  const v = [velocityVector[0], velocityVector[1]]; // RG channels for selected direction

  // prettier-ignore
  const velocityData = new Float32Array([
    _, _, _, _, _, _, v, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, v, v, v, _, _, _, _, _, _, _,
    _, v, _, _, _, _, _, v, v, v, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, v, v, v, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
  ].flat());
  device.queue.writeTexture(
    { texture: velocityTexture },
    velocityData,
    {
      bytesPerRow: GRID_SIZE * 2 * 4, // 2 channels × 4 bytes per 32-bit float
    },
    { width: GRID_SIZE, height: GRID_SIZE }
  );
}

type AdvectionTextureID = "velocity";

export class AdvectionPass extends ComputePass<SmokeTextureID> {
  constructor(device: GPUDevice) {
    super(
      {
        name: "Advection Pass",
        entryPoint: "compute_main",
        shader: device.createShaderModule({
          label: "Advection Shader",
          code: wgsl(SHADERS.ADVECTION, {
            variables: {
              WORKGROUP_SIZE,
              OUT_FORMAT: "rg32float",
            },
          }),
        }),
      },
      device
    );
  }

  protected createBindGroupLayout(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "Advection Bind Group Layout",
      entries: [
        // Velocity
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
          },
        },
        // Advection In
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            sampleType: "unfilterable-float",
            viewDimension: "2d",
          },
        },
        // Advection Out
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            format: "rg32float",
            access: "write-only",
            viewDimension: "2d",
          },
        },
        // Uniform Input Buffer
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

  protected createBindGroup(args: BindGroupArgs<SmokeTextureID>): GPUBindGroup {
    if (!args.textureManager) {
      throw new Error("No texture manager supplied");
    }
    if (!args.uniformBuffer) {
      throw new Error("No uniform buffer supplied");
    }
    // const uniformBuffer = this.device.createBuffer({
    //   label: `${this.config.name} Uniform Buffer`,
    //   usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    //   size: args.uniformValues.byteLength,
    // });
    // this.device.queue.writeBuffer(uniformBuffer, 0, args.uniformValues);
    return this.device.createBindGroup({
      label: "Advection Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: args.textureManager
            .getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 1,
          resource: args.textureManager
            .getCurrentTexture("velocity")
            .createView(),
        },
        {
          binding: 2,
          resource: args.textureManager.getBackTexture("velocity").createView(),
        },
        {
          binding: 3,
          resource: {
            buffer: args.uniformBuffer,
          },
        },
      ],
    });
  }
}

class AdvectionTestSimulation {
  private resources: WebGPUResources | null = null;
  private sampler: GPUSampler | null = null;
  private textureManager: TextureManager<AdvectionTextureID> | null = null;
  private advectionPass: AdvectionPass | null = null;
  private renderingPass: RenderPass<SmokeTextureID> | null = null;

  public async initialize(
    canvasRef: React.RefObject<HTMLCanvasElement>,
    direction: AdvectionDirection = AdvectionDirection.EAST
  ) {
    if (!canvasRef.current) {
      throw new WebGPUError(
        "Could not initialize WebGPU: Canvas not found",
        WebGPUErrorCode.NO_CANVAS
      );
    }
    this.resources = await initializeWebGPU(canvasRef.current);

    // Initialize texture manager
    this.textureManager = new TextureManager<AdvectionTextureID>(
      this.resources.device
    );

    // Create velocity texture (ping-pong for advection)
    this.textureManager.createPingPongTexture("velocity", {
      label: "Velocity Texture",
      size: [GRID_SIZE, GRID_SIZE],
      format: "rg32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    this.sampler = this.resources.device.createSampler({
      label: "Texture Sampler",
    });

    // Initialize velocity field with test pattern in the specified direction
    initializeVelocityField(
      this.textureManager.getCurrentTexture("velocity"),
      this.resources.device,
      direction
    );

    // Create advection compute pass
    this.advectionPass = new AdvectionPass(this.resources.device);

    // Create rendering pass
    const textureShaderModule = this.resources.device.createShaderModule({
      label: "Texture Shader",
      code: wgsl(SHADERS.TEXTURE),
    });
    this.renderingPass = new RenderPass<SmokeTextureID>(
      {
        name: "Advection Test Rendering",
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
      !this.advectionPass ||
      !this.renderingPass
    ) {
      throw new WebGPUError(
        "Could not step advection test: Resources not initialized. Run initialize() first.",
        WebGPUErrorCode.NO_RESOURCES
      );
    }

    const commandEncoder = this.resources.device.createCommandEncoder();

    if (!renderOnly) {
      // Create uniform values for the advection pass
      const uniformValues = new Float32Array([
        1.0 / 30.0, // timestep
        ADVECTION_FACTOR,
      ]);
      const uniformBuffer = this.resources.device.createBuffer({
        label: "Advection UBO",
        size: uniformValues.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.resources.device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

      // Compute Pass - Run Advection
      const advectionPassEncoder = commandEncoder.beginComputePass({
        label: "Advection Test Compute Pass",
      });
      this.advectionPass.execute(
        advectionPassEncoder,
        {
          uniformBuffer,
          textureManager: this.textureManager,
        },
        WORKGROUP_COUNT
      );
      advectionPassEncoder.end();

      // Swap textures for next iteration
      this.textureManager.swap("velocity");
    }

    // Render Pass - Display velocity field
    const renderPassEncoder = commandEncoder.beginRenderPass({
      label: "Advection Test Render Pass",
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
      smokeColor: [0.0, 1.0, 0.0],
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

function AdvectionTestComponent() {
  const advectionTestSimulation = useRef<AdvectionTestSimulation | null>(null);
  const [selectedDirection, setSelectedDirection] =
    useState<AdvectionDirection>(AdvectionDirection.EAST);

  const handleInitialize = async (
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => {
    if (!canvasRef.current || advectionTestSimulation.current) {
      return;
    }

    advectionTestSimulation.current = new AdvectionTestSimulation();
    await advectionTestSimulation.current.initialize(
      canvasRef,
      selectedDirection
    );
  };

  const handleStep = () => {
    if (advectionTestSimulation.current) {
      advectionTestSimulation.current.step();
    }
  };

  const handleRestart = async (
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => {
    if (advectionTestSimulation.current && canvasRef.current) {
      await advectionTestSimulation.current.initialize(
        canvasRef,
        selectedDirection
      );
    }
  };

  const handleDirectionChange = async (
    newDirection: AdvectionDirection,
    canvasRef: React.RefObject<HTMLCanvasElement>
  ) => {
    setSelectedDirection(newDirection);
    // Reinitialize with new direction
    if (advectionTestSimulation.current && canvasRef.current) {
      await advectionTestSimulation.current.initialize(canvasRef, newDirection);
    }
  };

  // Direction selector component
  const DirectionSelector = ({
    canvasRef,
  }: {
    canvasRef: React.RefObject<HTMLCanvasElement>;
  }) => (
    <div className="flex flex-col items-center gap-2 mb-4">
      <label
        htmlFor="direction-select"
        className="text-sm font-medium text-gray-700"
      >
        Advection Direction:
      </label>
      <select
        id="direction-select"
        value={selectedDirection}
        onChange={(e) =>
          handleDirectionChange(e.target.value as AdvectionDirection, canvasRef)
        }
        className="px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        {Object.values(AdvectionDirection).map((direction) => (
          <option key={direction} value={direction}>
            {direction} ({DIRECTION_VECTORS[direction].join(", ")})
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <BaseTestComponent
      title="Advection Test"
      onInitialize={handleInitialize}
      onStep={handleStep}
      onRestart={handleRestart}
      customControls={(canvasRef) => (
        <DirectionSelector canvasRef={canvasRef} />
      )}
    />
  );
}

export default AdvectionTestComponent;
