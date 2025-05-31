// Type-safe uniform management system for WebGPU compute passes

/**
 * Base interface for uniform data structures.
 * Supports both simple float arrays and complex mixed-type structures.
 */
export interface UniformData {
  readonly byteLength: number;

  /**
   * Simple method for basic float-only uniforms (backward compatibility).
   * If your uniform contains only floats, implement this method.
   */
  toFloat32Array?(): Float32Array;

  /**
   * Advanced method for complex uniforms with mixed data types.
   * This gives you full control over how data is written to the buffer.
   * Use this for structs with integers, vectors, matrices, or custom layouts.
   */
  writeToBuffer?(device: GPUDevice, buffer: GPUBuffer, offset?: number): void;

  /**
   * Returns data used for change detection and caching.
   * Should return consistent data for the same uniform values.
   */
  getHashData(): ArrayBuffer | ArrayBufferView | string;
}

/**
 * Utility class for complex uniform buffer writing with proper alignment.
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

  public writeVec2f(x: number, y: number): void {
    this.alignTo(8); // vec2 alignment
    this.writeFloat32(x);
    this.writeFloat32(y);
  }

  public writeVec3f(x: number, y: number, z: number): void {
    this.alignTo(16); // vec3 alignment (same as vec4)
    this.writeFloat32(x);
    this.writeFloat32(y);
    this.writeFloat32(z);
    this.offset += 4; // padding
  }

  public writeVec4f(x: number, y: number, z: number, w: number): void {
    this.alignTo(16); // vec4 alignment
    this.writeFloat32(x);
    this.writeFloat32(y);
    this.writeFloat32(z);
    this.writeFloat32(w);
  }

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
 * Manages a single uniform buffer with automatic sizing, reuse, and type safety.
 * Each pass should have its own UniformManager instance.
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
   * Gets or creates a uniform buffer for the given data.
   * Reuses the buffer if the data hasn't changed and size is compatible.
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

    // Write new data using the appropriate method
    this.writeDataToBuffer(data);
    this.lastDataHash = dataHash;

    return this.buffer;
  }

  /**
   * Forces a buffer update even if data hash matches.
   * Useful for dynamic data that may have same values but needs refresh.
   */
  public forceUpdate(data: T): GPUBuffer {
    this.lastDataHash = null;
    return this.getBuffer(data);
  }

  /**
   * Destroys the managed buffer. Should be called during cleanup.
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

    // Use custom writeToBuffer method if available
    if (data.writeToBuffer) {
      data.writeToBuffer(this.device, this.buffer, 0);
    }
    // Fallback to toFloat32Array for backward compatibility
    else if (data.toFloat32Array) {
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
 * Specific uniform data structures for different pass types.
 * These implement UniformData for type safety and automatic sizing.
 */

export class AdvectionUniforms implements UniformData {
  readonly byteLength = 2 * 4; // 2 floats

  constructor(
    public readonly timestep: number,
    public readonly advectionFactor: number
  ) {}

  toFloat32Array(): Float32Array {
    return new Float32Array([this.timestep, this.advectionFactor]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class DiffusionUniforms implements UniformData {
  readonly byteLength = 2 * 4; // 2 floats

  constructor(
    public readonly timestep: number,
    public readonly diffusionFactor: number
  ) {}

  toFloat32Array(): Float32Array {
    const alpha = 1 / this.diffusionFactor / this.timestep;
    const rBeta = 1.0 / (4.0 + alpha);
    return new Float32Array([alpha, rBeta]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class DivergenceUniforms implements UniformData {
  readonly byteLength = 1 * 4; // 1 float

  constructor(public readonly gridScale: number) {}

  toFloat32Array(): Float32Array {
    const halfRdx = 0.5 / this.gridScale;
    return new Float32Array([halfRdx]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class PressureUniforms implements UniformData {
  readonly byteLength = 2 * 4; // 2 floats

  constructor(public readonly gridScale: number) {}

  toFloat32Array(): Float32Array {
    const rdx = 1.0 / this.gridScale;
    const alpha = -(rdx * rdx);
    const rBeta = 0.25;
    return new Float32Array([alpha, rBeta]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class GradientSubtractionUniforms implements UniformData {
  readonly byteLength = 1 * 4; // 1 float

  constructor(public readonly gridScale: number) {}

  toFloat32Array(): Float32Array {
    const halfRdx = 0.5 / this.gridScale;
    return new Float32Array([halfRdx]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class BoundaryUniforms implements UniformData {
  readonly byteLength = 2 * 4; // 2 floats

  constructor(
    public readonly boundaryType: number,
    public readonly scale: number = 1.0
  ) {}

  toFloat32Array(): Float32Array {
    return new Float32Array([this.boundaryType, this.scale]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class AddSmokeUniforms implements UniformData {
  readonly byteLength = 4 * 4; // 4 floats

  constructor(
    public readonly positionX: number,
    public readonly positionY: number,
    public readonly radius: number,
    public readonly intensity: number
  ) {}

  toFloat32Array(): Float32Array {
    return new Float32Array([
      this.positionX,
      this.positionY,
      this.radius,
      this.intensity,
    ]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class AddVelocityUniforms implements UniformData {
  readonly byteLength = 6 * 4; // 6 floats

  constructor(
    public readonly positionX: number,
    public readonly positionY: number,
    public readonly velocityX: number,
    public readonly velocityY: number,
    public readonly radius: number,
    public readonly intensity: number
  ) {}

  toFloat32Array(): Float32Array {
    return new Float32Array([
      this.positionX,
      this.positionY,
      this.velocityX,
      this.velocityY,
      this.radius,
      this.intensity,
    ]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

export class DissipationUniforms implements UniformData {
  readonly byteLength = 1 * 4; // 1 float

  constructor(public readonly dissipationFactor: number) {}

  toFloat32Array(): Float32Array {
    return new Float32Array([this.dissipationFactor]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

// Example of complex uniform with mixed data types
export class ComplexShaderUniforms implements UniformData {
  readonly byteLength = 64; // Calculated based on WebGPU alignment rules

  constructor(
    public readonly viewMatrix: Float32Array, // mat4x4 - 64 bytes
    public readonly lightCount: number, // u32 - 4 bytes
    public readonly time: number, // f32 - 4 bytes
    public readonly cameraPosition: [number, number, number], // vec3 - 12 bytes + 4 padding
    public readonly flags: number // u32 - 4 bytes
  ) {
    if (viewMatrix.length !== 16) {
      throw new Error("View matrix must be 4x4 (16 elements)");
    }
  }

  // Use custom buffer writing for complex layout
  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    // Write mat4x4 (aligned to 16 bytes)
    writer.writeMat4x4f(this.viewMatrix);

    // Write u32 light count (aligned to 4 bytes)
    writer.writeUint32(this.lightCount);

    // Write f32 time (aligned to 4 bytes)
    writer.writeFloat32(this.time);

    // Write vec3 camera position (aligned to 16 bytes, with padding)
    writer.writeVec3f(...this.cameraPosition);

    // Write u32 flags (aligned to 4 bytes)
    writer.writeUint32(this.flags);

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): string {
    // Create a hash from all the data
    const parts = [
      Array.from(this.viewMatrix).join(","),
      this.lightCount.toString(),
      this.time.toString(),
      this.cameraPosition.join(","),
      this.flags.toString(),
    ];
    return parts.join("|");
  }
}

// Example of uniform with dynamic arrays
export class LightArrayUniforms implements UniformData {
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
