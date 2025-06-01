/**
 * Type-safe uniform management system for WebGPU compute passes.
 *
 * This system provides automatic buffer management, change detection, and caching
 * for uniform data in WebGPU compute shaders. It supports both simple float-only
 * uniforms and complex mixed-type structures with proper alignment.
 */

/**
 * Base interface for all uniform data structures.
 * Defines the minimum requirements for data that can be managed by UniformManager.
 */
interface BaseUniformData {
  readonly byteLength: number;

  /**
   * Returns data used for change detection and caching.
   * Should return consistent data for the same uniform values.
   * This is used to determine if the GPU buffer needs to be updated.
   */
  getHashData(): ArrayBuffer | ArrayBufferView | string;
}

/**
 * Interface for uniform data containing only float values.
 *
 * Use this for simple uniforms like timesteps, factors, and basic parameters
 * that can be efficiently represented as a Float32Array. This is the most
 * common case for compute shader uniforms.
 */
interface SimpleUniformData extends BaseUniformData {
  /**
   * Convert uniform data to a Float32Array for GPU upload.
   * This method should be efficient and avoid allocations when possible.
   */
  toFloat32Array(): Float32Array;
}

/**
 * Interface for uniform data with mixed data types or complex layouts.
 *
 * Use this when you need integers, vectors, matrices, or when you need
 * precise control over memory layout and alignment. This is necessary
 * for complex shader structs that don't map cleanly to Float32Array.
 */
interface ComplexUniformData extends BaseUniformData {
  /**
   * Write uniform data directly to a GPU buffer with full control.
   *
   * This method gives you complete control over memory layout, alignment,
   * and data types. Use UniformBufferWriter for complex layouts that need
   * to match WGSL struct alignment rules.
   */
  writeToBuffer(device: GPUDevice, buffer: GPUBuffer, offset?: number): void;
}

/**
 * Union type for uniform data structures.
 *
 * Supports both simple float arrays and complex mixed-type structures.
 * Each uniform class should implement exactly one of the two interfaces,
 * never both.
 */
export type UniformData = SimpleUniformData | ComplexUniformData;

/**
 * Utility for writing complex uniform data with proper WebGPU alignment.
 *
 * WebGPU has strict alignment requirements for uniform buffers that match
 * WGSL struct layout rules. This writer handles the alignment automatically
 * so you don't have to calculate offsets manually.
 *
 * Example:
 * ```ts
 * const data = new ArrayBuffer(1024);
 * const writer = new UniformBufferWriter(data);
 * writer.writeFloat32(1.0);        // Basic float
 * writer.writeVec3f(1, 2, 3);      // Vector with automatic padding
 * writer.writeUint32(42);          // Integer type
 * ```
 */
export class UniformBufferWriter {
  private view: DataView;
  private offset: number = 0;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  // Alignment helpers for WebGPU uniform buffer rules
  private alignTo(alignment: number): void {
    this.offset = Math.ceil(this.offset / alignment) * alignment;
  }

  public writeFloat32(value: number): void {
    this.alignTo(4);
    this.view.setFloat32(this.offset, value, true); // little-endian
    this.offset += 4;
  }

  public writeInt32(value: number): void {
    this.alignTo(4);
    this.view.setInt32(this.offset, value, true);
    this.offset += 4;
  }

  public writeUint32(value: number): void {
    this.alignTo(4);
    this.view.setUint32(this.offset, value, true);
    this.offset += 4;
  }

  /** Write a 2D vector with proper 8-byte alignment */
  public writeVec2f(x: number, y: number): void {
    this.alignTo(8); // vec2 alignment
    this.writeFloat32(x);
    this.writeFloat32(y);
  }

  /** Write a 3D vector with vec4 alignment and padding */
  public writeVec3f(x: number, y: number, z: number): void {
    this.alignTo(16); // vec3 alignment (same as vec4)
    this.writeFloat32(x);
    this.writeFloat32(y);
    this.writeFloat32(z);
    this.offset += 4; // padding
  }

  /** Write a 4D vector with 16-byte alignment */
  public writeVec4f(x: number, y: number, z: number, w: number): void {
    this.alignTo(16); // vec4 alignment
    this.writeFloat32(x);
    this.writeFloat32(y);
    this.writeFloat32(z);
    this.writeFloat32(w);
  }

  /** Write a 4x4 matrix with proper alignment */
  public writeMat4x4f(matrix: Float32Array | number[]): void {
    this.alignTo(16); // mat4x4 alignment
    if (matrix.length !== 16) {
      throw new Error("Matrix must have 16 elements");
    }
    for (let i = 0; i < 16; i++) {
      this.writeFloat32(matrix[i]);
    }
  }

  public writeFloat32Array(array: Float32Array | number[]): void {
    for (const value of array) {
      this.writeFloat32(value);
    }
  }

  public writeInt32Array(array: Int32Array | number[]): void {
    for (const value of array) {
      this.writeInt32(value);
    }
  }

  public writeUint32Array(array: Uint32Array | number[]): void {
    for (const value of array) {
      this.writeUint32(value);
    }
  }

  public getCurrentOffset(): number {
    return this.offset;
  }

  public skip(bytes: number): void {
    this.offset += bytes;
  }

  public padTo(alignment: number): void {
    this.alignTo(alignment);
  }
}

/**
 * Manages uniform buffers with automatic caching and change detection.
 *
 * Each compute pass should have its own UniformManager instance. The manager
 * handles buffer creation, resizing, and caching to avoid unnecessary GPU
 * uploads when uniform data hasn't changed.
 *
 * Features:
 * - Smart caching based on data content hashing
 * - Automatic buffer resizing when needed
 * - Proper resource cleanup
 * - Type safety through generics
 */
export class UniformManager<T extends UniformData> {
  private buffer: GPUBuffer | null = null;
  private lastDataHash: string | null = null;

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    private readonly usage: GPUBufferUsageFlags = GPUBufferUsage.UNIFORM |
      GPUBufferUsage.COPY_DST
  ) {}

  /**
   * Get a uniform buffer for the given data, creating or reusing as needed.
   *
   * This method is the main entry point for getting uniform buffers. It will:
   * 1. Check if the data has changed using hash comparison
   * 2. Reuse the existing buffer if data is unchanged and size fits
   * 3. Create a new buffer if needed or resize if too small
   * 4. Upload the data to the buffer
   *
   * The returned buffer is ready to bind to a compute pass.
   */
  public getBuffer(data: T): GPUBuffer {
    const dataHash = this.getDataHash(data);

    // Check if we can reuse the existing buffer
    if (
      this.buffer &&
      this.buffer.size >= data.byteLength &&
      this.lastDataHash === dataHash
    ) {
      return this.buffer;
    }

    // Create new buffer if needed or resize
    if (!this.buffer || this.buffer.size < data.byteLength) {
      this.buffer?.destroy();
      this.buffer = this.device.createBuffer({
        label: `${this.label} Uniform Buffer`,
        size: Math.max(data.byteLength, 256), // Minimum size to reduce reallocations
        usage: this.usage,
      });
    }

    this.writeDataToBuffer(data);
    this.lastDataHash = dataHash;

    return this.buffer;
  }

  /**
   * Clean up GPU resources.
   * Should be called when the manager is no longer needed.
   */
  public destroy(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.lastDataHash = null;
  }

  private writeDataToBuffer(data: T): void {
    if (!this.buffer) {
      throw new Error("Buffer not initialized");
    }

    // Use custom writeToBuffer method if available (ComplexUniformData)
    if ("writeToBuffer" in data) {
      data.writeToBuffer(this.device, this.buffer, 0);
    }
    // Fallback to toFloat32Array for simple float-only uniforms (SimpleUniformData)
    else if ("toFloat32Array" in data) {
      const floatArray = data.toFloat32Array();
      this.device.queue.writeBuffer(this.buffer, 0, floatArray);
    } else {
      throw new Error(
        `UniformData must implement either writeToBuffer() or toFloat32Array(). ` +
          `Got: ${Object.getOwnPropertyNames(data)}`
      );
    }
  }

  private getDataHash(data: T): string {
    const hashData = data.getHashData();

    if (typeof hashData === "string") {
      return hashData;
    }

    // Handle ArrayBuffer or ArrayBufferView
    const bytes =
      hashData instanceof ArrayBuffer
        ? new Uint8Array(hashData)
        : new Uint8Array(
            hashData.buffer,
            hashData.byteOffset,
            hashData.byteLength
          );

    // Simple hash function for bytes
    let hash = 0;
    for (let i = 0; i < bytes.length; i++) {
      hash = ((hash << 5) - hash + bytes[i]) & 0xffffffff;
    }
    return hash.toString();
  }
}

/**
 * Predefined uniform data classes for common fluid simulation passes.
 *
 * These classes implement the UniformData interface and provide type-safe
 * uniform management for specific compute passes. Each class handles the
 * specific data layout and calculations needed for its corresponding shader.
 */

/**
 * Uniform data for advection passes (velocity and scalar transport).
 * Contains timestep and advection strength parameters.
 */
export class AdvectionUniforms implements SimpleUniformData {
  readonly byteLength = 2 * 4; // 2 floats
  private static readonly reusableArray = new Float32Array(2);

  constructor(
    public readonly timestep: number,
    public readonly advectionFactor: number
  ) {}

  toFloat32Array(): Float32Array {
    AdvectionUniforms.reusableArray[0] = this.timestep;
    AdvectionUniforms.reusableArray[1] = this.advectionFactor;
    return AdvectionUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for diffusion passes (Jacobi iteration parameters).
 * Automatically calculates alpha and rBeta values from timestep and diffusion factor.
 */
export class DiffusionUniforms implements SimpleUniformData {
  readonly byteLength = 2 * 4; // 2 floats
  private static readonly reusableArray = new Float32Array(2);

  constructor(
    public readonly timestep: number,
    public readonly diffusionFactor: number
  ) {}

  toFloat32Array(): Float32Array {
    const alpha = 1 / this.diffusionFactor / this.timestep;
    const rBeta = 1.0 / (4.0 + alpha);
    DiffusionUniforms.reusableArray[0] = alpha;
    DiffusionUniforms.reusableArray[1] = rBeta;
    return DiffusionUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for divergence computation.
 * Contains the grid scale factor for finite difference calculations.
 */
export class DivergenceUniforms implements SimpleUniformData {
  readonly byteLength = 1 * 4; // 1 float
  private static readonly reusableArray = new Float32Array(1);

  constructor(public readonly gridScale: number) {}

  toFloat32Array(): Float32Array {
    const halfRdx = 0.5 / this.gridScale;
    DivergenceUniforms.reusableArray[0] = halfRdx;
    return DivergenceUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for pressure solving (Jacobi iteration for Poisson equation).
 * Calculates alpha and rBeta parameters for the pressure Poisson equation.
 */
export class PressureUniforms implements SimpleUniformData {
  readonly byteLength = 2 * 4; // 2 floats
  private static readonly reusableArray = new Float32Array(2);

  constructor(public readonly gridScale: number) {}

  toFloat32Array(): Float32Array {
    const rdx = 1.0 / this.gridScale;
    const alpha = -(rdx * rdx);
    const rBeta = 0.25;
    PressureUniforms.reusableArray[0] = alpha;
    PressureUniforms.reusableArray[1] = rBeta;
    return PressureUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for gradient subtraction (pressure projection).
 * Contains grid scale for computing pressure gradient.
 */
export class GradientSubtractionUniforms implements SimpleUniformData {
  readonly byteLength = 1 * 4; // 1 float
  private static readonly reusableArray = new Float32Array(1);

  constructor(public readonly gridScale: number) {}

  toFloat32Array(): Float32Array {
    const halfRdx = 0.5 / this.gridScale;
    GradientSubtractionUniforms.reusableArray[0] = halfRdx;
    return GradientSubtractionUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for boundary conditions.
 * Specifies the type of boundary condition and optional scale factor.
 */
export class BoundaryUniforms implements SimpleUniformData {
  readonly byteLength = 2 * 4; // 2 floats
  private static readonly reusableArray = new Float32Array(2);

  constructor(
    public readonly boundaryType: number,
    public readonly scale: number = 1.0
  ) {}

  toFloat32Array(): Float32Array {
    BoundaryUniforms.reusableArray[0] = this.boundaryType;
    BoundaryUniforms.reusableArray[1] = this.scale;
    return BoundaryUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for adding smoke at a specific location.
 * Contains position, radius, and intensity parameters for interactive smoke injection.
 */
export class AddSmokeUniforms implements SimpleUniformData {
  readonly byteLength = 4 * 4; // 4 floats
  private static readonly reusableArray = new Float32Array(4);

  constructor(
    public readonly positionX: number,
    public readonly positionY: number,
    public readonly radius: number,
    public readonly intensity: number
  ) {}

  toFloat32Array(): Float32Array {
    AddSmokeUniforms.reusableArray[0] = this.positionX;
    AddSmokeUniforms.reusableArray[1] = this.positionY;
    AddSmokeUniforms.reusableArray[2] = this.radius;
    AddSmokeUniforms.reusableArray[3] = this.intensity;
    return AddSmokeUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for adding velocity based on mouse interaction.
 * Contains position, velocity direction, radius, and intensity for fluid stirring.
 */
export class AddVelocityUniforms implements SimpleUniformData {
  readonly byteLength = 6 * 4; // 6 floats
  private static readonly reusableArray = new Float32Array(6);

  constructor(
    public readonly positionX: number,
    public readonly positionY: number,
    public readonly velocityX: number,
    public readonly velocityY: number,
    public readonly radius: number,
    public readonly intensity: number
  ) {}

  toFloat32Array(): Float32Array {
    AddVelocityUniforms.reusableArray[0] = this.positionX;
    AddVelocityUniforms.reusableArray[1] = this.positionY;
    AddVelocityUniforms.reusableArray[2] = this.velocityX;
    AddVelocityUniforms.reusableArray[3] = this.velocityY;
    AddVelocityUniforms.reusableArray[4] = this.radius;
    AddVelocityUniforms.reusableArray[5] = this.intensity;
    return AddVelocityUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Uniform data for dissipation passes.
 * Contains a single dissipation factor for gradually reducing field values over time.
 */
export class DissipationUniforms implements SimpleUniformData {
  readonly byteLength = 1 * 4; // 1 float
  private static readonly reusableArray = new Float32Array(1);

  constructor(public readonly dissipationFactor: number) {}

  toFloat32Array(): Float32Array {
    DissipationUniforms.reusableArray[0] = this.dissipationFactor;
    return DissipationUniforms.reusableArray;
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

/**
 * Example of complex uniform data with dynamic arrays and mixed types.
 *
 * This demonstrates how to use ComplexUniformData for advanced scenarios
 * like lighting systems with variable numbers of lights and proper alignment.
 * Most fluid simulation passes won't need this complexity.
 */
export class LightArrayUniforms implements ComplexUniformData {
  readonly byteLength: number;

  constructor(
    public readonly lights: Array<{
      position: [number, number, number];
      color: [number, number, number];
      intensity: number;
    }>,
    public readonly ambientColor: [number, number, number]
  ) {
    // Each light: vec3 position + vec3 color + f32 intensity = 7 floats = 28 bytes
    // But due to alignment, each light takes 32 bytes (vec3 aligns to 16)
    // Ambient color: vec3 = 12 bytes + 4 padding = 16 bytes
    // Light count: u32 = 4 bytes + 12 padding = 16 bytes (to align next struct)
    this.byteLength = 16 + 16 + lights.length * 32;
  }

  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    // Write light count
    writer.writeUint32(this.lights.length);
    writer.padTo(16); // Align to next struct

    // Write ambient color
    writer.writeVec3f(...this.ambientColor);
    writer.padTo(16); // Align to next struct

    // Write each light
    for (const light of this.lights) {
      writer.writeVec3f(...light.position);
      writer.writeVec3f(...light.color);
      writer.writeFloat32(light.intensity);
      writer.padTo(32); // Align to next light struct
    }

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): string {
    const lightData = this.lights
      .map(
        (light) =>
          `${light.position.join(",")};${light.color.join(",")};${light.intensity}`
      )
      .join("|");
    return `${this.lights.length};${this.ambientColor.join(",")};${lightData}`;
  }
}
