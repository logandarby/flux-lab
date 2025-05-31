/**
 * Examples demonstrating the enhanced UniformManager capabilities
 * for both simple and complex uniform structures.
 */

import { UniformManager, UniformBufferWriter } from "../utils/UniformManager";
import type { UniformData } from "../utils/UniformManager";

// ===== SIMPLE UNIFORMS (Float-only) =====

/**
 * Simple uniform - just use toFloat32Array() for backward compatibility
 */
export class SimplePhysicsUniforms implements UniformData {
  readonly byteLength = 3 * 4; // 3 floats

  constructor(
    public readonly gravity: number,
    public readonly damping: number,
    public readonly deltaTime: number
  ) {}

  toFloat32Array(): Float32Array {
    return new Float32Array([this.gravity, this.damping, this.deltaTime]);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array();
  }
}

// ===== MIXED DATA TYPE UNIFORMS =====

/**
 * Uniform with mixed integers and floats
 */
export class ParticleSystemUniforms implements UniformData {
  readonly byteLength = 32; // Calculated based on alignment

  constructor(
    public readonly particleCount: number, // u32
    public readonly maxAge: number, // f32
    public readonly gravity: [number, number], // vec2f
    public readonly emitterPosition: [number, number, number], // vec3f
    public readonly isActive: boolean // bool -> u32
  ) {}

  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    writer.writeUint32(this.particleCount);
    writer.writeFloat32(this.maxAge);
    writer.writeVec2f(...this.gravity);
    writer.writeVec3f(...this.emitterPosition);
    writer.writeUint32(this.isActive ? 1 : 0);

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): string {
    return [
      this.particleCount,
      this.maxAge,
      ...this.gravity,
      ...this.emitterPosition,
      this.isActive ? 1 : 0,
    ].join(",");
  }
}

/**
 * Advanced rendering uniforms with matrices and multiple data types
 */
export class CameraUniforms implements UniformData {
  readonly byteLength = 144; // mat4 + mat4 + vec3 + f32 + u32 with padding

  constructor(
    public readonly viewMatrix: Float32Array, // mat4x4f
    public readonly projectionMatrix: Float32Array, // mat4x4f
    public readonly cameraPosition: [number, number, number], // vec3f
    public readonly nearPlane: number, // f32
    public readonly renderMode: number // u32 (enum)
  ) {
    if (viewMatrix.length !== 16 || projectionMatrix.length !== 16) {
      throw new Error("Matrices must be 4x4 (16 elements each)");
    }
  }

  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    // Matrices are aligned to 16 bytes
    writer.writeMat4x4f(this.viewMatrix);
    writer.writeMat4x4f(this.projectionMatrix);

    // vec3 is aligned to 16 bytes
    writer.writeVec3f(...this.cameraPosition);

    // Additional scalar values
    writer.writeFloat32(this.nearPlane);
    writer.writeUint32(this.renderMode);

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): string {
    return [
      Array.from(this.viewMatrix).join(","),
      Array.from(this.projectionMatrix).join(","),
      this.cameraPosition.join(","),
      this.nearPlane,
      this.renderMode,
    ].join("|");
  }
}

// ===== DYNAMIC ARRAY UNIFORMS =====

/**
 * Uniform with dynamic array of data
 */
export class MaterialArrayUniforms implements UniformData {
  readonly byteLength: number;

  constructor(
    public readonly materials: Array<{
      diffuseColor: [number, number, number];
      specularColor: [number, number, number];
      roughness: number;
      metallic: number;
    }>
  ) {
    // Each material: 2 vec3 + 2 floats = 8 floats = 32 bytes
    // Material count: u32 = 4 bytes + 12 padding = 16 bytes
    this.byteLength = 16 + materials.length * 32;
  }

  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    // Write material count
    writer.writeUint32(this.materials.length);
    writer.padTo(16); // Align to next struct

    // Write each material
    for (const material of this.materials) {
      writer.writeVec3f(...material.diffuseColor);
      writer.writeFloat32(material.roughness); // This fills the vec3 padding
      writer.writeVec3f(...material.specularColor);
      writer.writeFloat32(material.metallic); // This fills the vec3 padding
    }

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): string {
    const materialData = this.materials
      .map(
        (mat) =>
          `${mat.diffuseColor.join(",")};${mat.specularColor.join(",")};${mat.roughness};${mat.metallic}`
      )
      .join("|");
    return `${this.materials.length};${materialData}`;
  }
}

// ===== SHADER VARIANT UNIFORMS =====

/**
 * Uniform that changes structure based on a variant/mode
 */
export class AdaptiveShaderUniforms implements UniformData {
  readonly byteLength: number;

  constructor(
    public readonly mode: "basic" | "advanced" | "debug",
    public readonly baseColor: [number, number, number],
    public readonly options?: {
      // Only used in 'advanced' mode
      lightingModel?: number;
      shadowQuality?: number;
      // Only used in 'debug' mode
      debugFlags?: number;
      wireframeColor?: [number, number, number];
    }
  ) {
    // Calculate size based on mode
    this.byteLength =
      mode === "basic"
        ? 16 // vec3 + u32 mode
        : mode === "advanced"
          ? 32 // + 2 u32s + padding
          : 48; // + u32 + vec3 + padding
  }

  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    // Common data
    const modeValue =
      this.mode === "basic" ? 0 : this.mode === "advanced" ? 1 : 2;
    writer.writeUint32(modeValue);
    writer.writeVec3f(...this.baseColor);

    // Mode-specific data
    if (this.mode === "advanced" && this.options) {
      writer.writeUint32(this.options.lightingModel ?? 0);
      writer.writeUint32(this.options.shadowQuality ?? 1);
    } else if (this.mode === "debug" && this.options) {
      writer.writeUint32(this.options.debugFlags ?? 0);
      writer.writeVec3f(...(this.options.wireframeColor ?? [1, 0, 0]));
    }

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): string {
    const baseHash = `${this.mode};${this.baseColor.join(",")}`;
    if (this.mode === "advanced" && this.options) {
      return `${baseHash};${this.options.lightingModel};${this.options.shadowQuality}`;
    } else if (this.mode === "debug" && this.options) {
      return `${baseHash};${this.options.debugFlags};${this.options.wireframeColor?.join(",")}`;
    }
    return baseHash;
  }
}

// ===== USAGE EXAMPLES =====

export class UniformUsageExamples {
  private device: GPUDevice;
  private simpleUniformManager: UniformManager<SimplePhysicsUniforms>;
  private complexUniformManager: UniformManager<CameraUniforms>;
  private dynamicUniformManager: UniformManager<MaterialArrayUniforms>;

  constructor(device: GPUDevice) {
    this.device = device;

    // Create managers for different uniform types
    this.simpleUniformManager = new UniformManager(device, "Simple Physics");
    this.complexUniformManager = new UniformManager(device, "Camera");
    this.dynamicUniformManager = new UniformManager(device, "Materials");
  }

  // Example 1: Simple uniform usage
  public updatePhysics(
    gravity: number,
    damping: number,
    deltaTime: number
  ): GPUBuffer {
    const uniforms = new SimplePhysicsUniforms(gravity, damping, deltaTime);
    return this.simpleUniformManager.getBuffer(uniforms);
  }

  // Example 2: Complex uniform with matrices
  public updateCamera(
    viewMatrix: Float32Array,
    projMatrix: Float32Array,
    position: [number, number, number],
    nearPlane: number,
    renderMode: number
  ): GPUBuffer {
    const uniforms = new CameraUniforms(
      viewMatrix,
      projMatrix,
      position,
      nearPlane,
      renderMode
    );
    return this.complexUniformManager.getBuffer(uniforms);
  }

  // Example 3: Dynamic array uniform
  public updateMaterials(
    materials: Array<{
      diffuseColor: [number, number, number];
      specularColor: [number, number, number];
      roughness: number;
      metallic: number;
    }>
  ): GPUBuffer {
    const uniforms = new MaterialArrayUniforms(materials);
    return this.dynamicUniformManager.getBuffer(uniforms);
  }

  // Example 4: Performance optimization - buffer reuse
  public demonstrateBufferReuse(): void {
    const uniforms1 = new SimplePhysicsUniforms(9.8, 0.99, 0.016);
    const uniforms2 = new SimplePhysicsUniforms(9.8, 0.99, 0.016); // Same values

    const buffer1 = this.simpleUniformManager.getBuffer(uniforms1);
    const buffer2 = this.simpleUniformManager.getBuffer(uniforms2);

    console.log("Buffers are same object:", buffer1 === buffer2); // true - buffer reused!
  }

  // Example 5: Manual buffer update for frequently changing data
  public forceUniformUpdate(
    gravity: number,
    damping: number,
    deltaTime: number
  ): GPUBuffer {
    const uniforms = new SimplePhysicsUniforms(gravity, damping, deltaTime);
    // Force update even if values might be the same
    return this.simpleUniformManager.forceUpdate(uniforms);
  }

  // Cleanup
  public destroy(): void {
    this.simpleUniformManager.destroy();
    this.complexUniformManager.destroy();
    this.dynamicUniformManager.destroy();
  }
}

// ===== MIGRATION EXAMPLES =====

/**
 * Example showing how to migrate from old toFloat32Array to new writeToBuffer
 */
export class MigrationExample implements UniformData {
  readonly byteLength = 20; // 4 floats + 1 u32

  constructor(
    public readonly values: [number, number, number, number],
    public readonly count: number
  ) {}

  // Old way (still supported for backward compatibility)
  toFloat32Array(): Float32Array {
    return new Float32Array([...this.values, this.count]);
  }

  // New way (more control, mixed data types)
  writeToBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    offset: number = 0
  ): void {
    const data = new ArrayBuffer(this.byteLength);
    const writer = new UniformBufferWriter(data);

    writer.writeVec4f(...this.values); // Write as vec4 (proper alignment)
    writer.writeUint32(this.count); // Write as uint32 (not float)

    device.queue.writeBuffer(buffer, offset, data);
  }

  getHashData(): Float32Array {
    return this.toFloat32Array(); // Can use either method for hashing
  }
}
