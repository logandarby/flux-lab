import type { GOLEngine as ISmokeEngine, GOLConfig, WebGPUResources } from './types';
import { 
  initializeWebGPU, 
  createVertexBuffer, 
  createUniformBuffer, 
  createStorageBuffers,
  createBindGroupLayout,
  createBindGroups,
  WebGPUError
} from './webgpu-utils';
import { createSimulationShaderCode, SQUARE_VERTICES, VERTEX_BUFFER_LAYOUT } from './shaders';
import cellShaderCode from '../cellShader.wgsl?raw';

export class SmokeEngine implements ISmokeEngine {
  private resources: WebGPUResources | null = null;
  private config: GOLConfig | null = null;
  private animationId: number | null = null;
  private lastUpdateTime = 0;
  private currentStep = 0;
  private running = false;
  private destroyed = false;

  async initialize(canvas: HTMLCanvasElement, config: GOLConfig): Promise<void> {
    if (this.destroyed) {
      throw new WebGPUError("Cannot initialize a destroyed engine. Create a new instance.");
    }

    if (this.resources) {
      throw new WebGPUError("Engine is already initialized. Call destroy() first.");
    }

    try {
      this.config = { ...config }; // Create a copy to avoid external mutations
      
      // Initialize WebGPU
      const { device, context, canvasFormat } = await initializeWebGPU(canvas);

      // Create buffers
      const vertexBuffer = createVertexBuffer(device, SQUARE_VERTICES);
      const uniformBuffer = createUniformBuffer(device, config.gridSize);
      const cellStateStorageBuffers = createStorageBuffers(device, config.gridSize);

      // Create shaders
      const simulationShaderModule = device.createShaderModule({
        label: "Simulation Compute Shader",
        code: createSimulationShaderCode(config.workgroupSize),
      });

      const cellShaderModule = device.createShaderModule({
        label: "Cell Shader",
        code: cellShaderCode,
      });

      // Create bind group layout and bind groups
      const bindGroupLayout = createBindGroupLayout(device);
      const bindGroups = createBindGroups(device, bindGroupLayout, uniformBuffer, cellStateStorageBuffers);

      // Create pipeline layout
      const pipelineLayout = device.createPipelineLayout({
        label: "Pipeline Layout (Cell and Simulation)",
        bindGroupLayouts: [bindGroupLayout],
      });

      // Create render pipeline
      const cellPipeline = device.createRenderPipeline({
        label: "Cell Pipeline",
        layout: pipelineLayout,
        vertex: {
          module: cellShaderModule,
          entryPoint: "vertexMain",
          buffers: [VERTEX_BUFFER_LAYOUT],
        },
        fragment: {
          module: cellShaderModule,
          entryPoint: "fragmentMain",
          targets: [{
            format: canvasFormat,
          }]
        }
      });

      // Create compute pipeline
      const simulationPipeline = device.createComputePipeline({
        label: "Simulation Pipeline",
        layout: pipelineLayout,
        compute: {
          module: simulationShaderModule,
          entryPoint: "computeMain",
        }
      });

      // Store all resources only if we haven't been destroyed during initialization
      if (!this.destroyed) {
        this.resources = {
          device,
          context,
          vertexBuffer,
          uniformBuffer,
          cellStateStorageBuffers,
          bindGroups,
          cellPipeline,
          simulationPipeline,
        };
      } else {
        // Clean up if we were destroyed during initialization
        vertexBuffer.destroy();
        uniformBuffer.destroy();
        cellStateStorageBuffers.forEach(buffer => buffer.destroy());
        throw new WebGPUError("Engine was destroyed during initialization");
      }

    } catch (error) {
      this.resources = null;
      this.config = null;
      
      if (error instanceof WebGPUError) {
        throw error;
      }
      throw new WebGPUError(`Failed to initialize WebGPU: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  start(): void {
    if (this.destroyed) {
      throw new WebGPUError("Cannot start a destroyed engine");
    }

    if (!this.resources || !this.config) {
      throw new WebGPUError("Engine not initialized. Call initialize() first.");
    }

    if (this.running) {
      return;
    }

    this.running = true;
    this.lastUpdateTime = performance.now();
    this.animate();
  }

  stop(): void {
    this.running = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  step(): void {
    if (this.destroyed) {
      throw new WebGPUError("Cannot step a destroyed engine");
    }

    if (!this.resources || !this.config) {
      throw new WebGPUError("Engine not initialized. Call initialize() first.");
    }

    this.updateGrid();
  }

  isRunning(): boolean {
    return this.running && !this.destroyed;
  }

  destroy(): void {
    if (this.destroyed) {
      return; // Already destroyed
    }

    this.destroyed = true;
    this.stop();
    
    if (this.resources) {
      try {
        // Clean up GPU resources
        this.resources.vertexBuffer.destroy();
        this.resources.uniformBuffer.destroy();
        this.resources.cellStateStorageBuffers.forEach(buffer => {
          try {
            buffer.destroy();
          } catch (err) {
            console.warn('Error destroying buffer:', err);
          }
        });
      } catch (err) {
        console.warn('Error during resource cleanup:', err);
      }
      
      // Note: Pipelines and bind groups don't have explicit destroy methods
      // They will be garbage collected when no longer referenced
      
      this.resources = null;
    }
    
    this.config = null;
    this.currentStep = 0;
  }

  private animate = (): void => {
    if (!this.running || !this.config || this.destroyed) {
      return;
    }

    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastUpdateTime;

    if (deltaTime >= this.config.updateIntervalMs) {
      try {
        this.updateGrid();
        this.lastUpdateTime = currentTime;
      } catch (err) {
        console.error('Error during grid update:', err);
        this.stop();
        return;
      }
    }

    if (this.running && !this.destroyed) {
      this.animationId = requestAnimationFrame(this.animate);
    }
  };

  private updateGrid(): void {
    if (!this.resources || !this.config || this.destroyed) {
      return;
    }

    const { device, context, vertexBuffer, bindGroups, cellPipeline, simulationPipeline } = this.resources;

    try {
      const encoder = device.createCommandEncoder();
      
      // Compute pass for grid simulation
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(simulationPipeline);
      computePass.setBindGroup(0, bindGroups[this.currentStep % 2]);
      const workgroupCount = Math.ceil(this.config.gridSize / this.config.workgroupSize);
      computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
      computePass.end();

      // Increment simulation step
      this.currentStep++;

      // Render pass
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store", 
          clearValue: { r: 0, g: 0, b: 0.4, a: 1 }
        }]
      });

      pass.setPipeline(cellPipeline);
      pass.setBindGroup(0, bindGroups[this.currentStep % 2]);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(SQUARE_VERTICES.length / 2, this.config.gridSize * this.config.gridSize);
      pass.end();

      // Submit commands to GPU
      device.queue.submit([encoder.finish()]);
    } catch (err) {
      console.error('Error during updateGrid:', err);
      throw new WebGPUError(`Failed to update grid: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }
} 