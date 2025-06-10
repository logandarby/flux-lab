/**
 * A 2D array utility class for storing texture data with proper dimensions
 */
export class Array2D<T> {
  private data: T[];
  public readonly width: number;
  public readonly height: number;

  constructor(width: number, height: number, initialValue?: T) {
    this.width = width;
    this.height = height;
    this.data = new Array(width * height);

    if (initialValue !== undefined) {
      this.data.fill(initialValue);
    }
  }

  /**
   * Create Array2D from a flat array
   */
  static fromArray<T>(data: T[], width: number, height: number): Array2D<T> {
    if (data.length !== width * height) {
      throw new Error(
        `Data length ${data.length} does not match dimensions ${width}x${height}`
      );
    }

    const array2D = new Array2D<T>(width, height);
    array2D.data = [...data];
    return array2D;
  }

  /**
   * Get value at (x, y) coordinates
   */
  get(x: number, y: number): T {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      throw new Error(
        `Coordinates (${x}, ${y}) out of bounds for ${this.width}x${this.height} array`
      );
    }
    return this.data[y * this.width + x];
  }

  /**
   * Set value at (x, y) coordinates
   */
  set(x: number, y: number, value: T): void {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      throw new Error(
        `Coordinates (${x}, ${y}) out of bounds for ${this.width}x${this.height} array`
      );
    }
    this.data[y * this.width + x] = value;
  }

  /**
   * Get value at flat array index
   */
  getIndex(index: number): T {
    if (index < 0 || index >= this.data.length) {
      throw new Error(
        `Index ${index} out of bounds for array of length ${this.data.length}`
      );
    }
    return this.data[index];
  }

  /**
   * Set value at flat array index
   */
  setIndex(index: number, value: T): void {
    if (index < 0 || index >= this.data.length) {
      throw new Error(
        `Index ${index} out of bounds for array of length ${this.data.length}`
      );
    }
    this.data[index] = value;
  }

  /**
   * Get the total number of elements
   */
  get length(): number {
    return this.data.length;
  }

  /**
   * Iterate over all values with their coordinates
   */
  forEach(
    callback: (value: T, x: number, y: number, index: number) => void
  ): void {
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const index = y * this.width + x;
        callback(this.data[index], x, y, index);
      }
    }
  }

  /**
   * Map over all values with their coordinates
   */
  map<U>(
    callback: (value: T, x: number, y: number, index: number) => U
  ): Array2D<U> {
    const result = new Array2D<U>(this.width, this.height);
    this.forEach((value, x, y, index) => {
      result.setIndex(index, callback(value, x, y, index));
    });
    return result;
  }

  /**
   * Fill the array with a value
   */
  fill(value: T): void {
    this.data.fill(value);
  }

  /**
   * Clone the array
   */
  clone(): Array2D<T> {
    return Array2D.fromArray(this.data, this.width, this.height);
  }

  /**
   * Get value at normalized coordinates (0-1 range)
   * @param normalizedX - X coordinate in range [0, 1]
   * @param normalizedY - Y coordinate in range [0, 1]
   * @returns The value at the corresponding grid position
   */
  getAtNormalized(normalizedX: number, normalizedY: number): T {
    if (
      normalizedX < 0 ||
      normalizedX > 1 ||
      normalizedY < 0 ||
      normalizedY > 1
    ) {
      throw new Error(
        `Normalized coordinates (${normalizedX}, ${normalizedY}) must be in range [0, 1]`
      );
    }

    const x = Math.floor(normalizedX * this.width);
    const y = Math.floor(normalizedY * this.height);

    // Clamp to valid indices (handle edge case where normalizedX/Y = 1.0)
    const clampedX = Math.min(x, this.width - 1);
    const clampedY = Math.min(y, this.height - 1);

    return this.get(clampedX, clampedY);
  }
}

/**
 * Specialized Array2D types for smoke simulation data
 */
export type ScalarField2D = Array2D<number>;
export type VectorField2D = Array2D<[number, number]>;

/**
 * Factory functions for creating downsampled texture arrays
 */
export function createScalarField2D(
  data: Float32Array,
  originalWidth: number,
  originalHeight: number,
  downSample: number = 1
): ScalarField2D {
  const sampledWidth = Math.floor(originalWidth / downSample);
  const sampledHeight = Math.floor(originalHeight / downSample);

  if (data.length !== sampledWidth * sampledHeight) {
    throw new Error(
      `Data length ${data.length} does not match expected downsampled dimensions ${sampledWidth}x${sampledHeight}`
    );
  }

  return Array2D.fromArray(Array.from(data), sampledWidth, sampledHeight);
}

export function createVectorField2D(
  data: Array<[number, number]>,
  originalWidth: number,
  originalHeight: number,
  downSample: number = 1
): VectorField2D {
  const sampledWidth = Math.floor(originalWidth / downSample);
  const sampledHeight = Math.floor(originalHeight / downSample);

  if (data.length !== sampledWidth * sampledHeight) {
    throw new Error(
      `Data length ${data.length} does not match expected downsampled dimensions ${sampledWidth}x${sampledHeight}`
    );
  }

  return Array2D.fromArray(data, sampledWidth, sampledHeight);
}
