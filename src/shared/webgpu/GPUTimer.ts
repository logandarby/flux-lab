/**
 * GPUTimer - Handles WebGPU timestamp queries for performance measurement
 */
export class GPUTimer {
  private querySet: GPUQuerySet | null = null;
  private resolveBuffer: GPUBuffer | null = null;
  private resultBuffer: GPUBuffer | null = null;
  private canTimestamp: boolean;
  private lastGpuTime: number = 0;

  constructor(device: GPUDevice, canTimestamp: boolean) {
    this.canTimestamp = canTimestamp;

    if (canTimestamp) {
      // Create query set for timestamp queries
      this.querySet = device.createQuerySet({
        type: "timestamp",
        count: 2, // Start and end timestamps
      });

      // Buffer to resolve query results
      this.resolveBuffer = device.createBuffer({
        size: this.querySet.count * 8, // 8 bytes per timestamp
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      // Mappable buffer to read results in JavaScript
      this.resultBuffer = device.createBuffer({
        size: this.resolveBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  /**
   * Get timestamp writes configuration for render/compute passes
   */
  getTimestampWrites():
    | GPUComputePassTimestampWrites
    | GPURenderPassTimestampWrites
    | undefined {
    if (!this.canTimestamp || !this.querySet) {
      return undefined;
    }

    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }

  /**
   * Resolve timestamp queries and copy to result buffer
   */
  resolveTimestamps(encoder: GPUCommandEncoder): void {
    if (
      !this.canTimestamp ||
      !this.querySet ||
      !this.resolveBuffer ||
      !this.resultBuffer
    ) {
      return;
    }

    // Resolve query set to buffer
    encoder.resolveQuerySet(
      this.querySet,
      0,
      this.querySet.count,
      this.resolveBuffer,
      0
    );

    // Copy to result buffer if it's not currently mapped
    if (this.resultBuffer.mapState === "unmapped") {
      encoder.copyBufferToBuffer(
        this.resolveBuffer,
        0,
        this.resultBuffer,
        0,
        this.resultBuffer.size
      );
    }
  }

  /**
   * Read GPU timing results asynchronously
   */
  async readResults(): Promise<void> {
    if (!this.canTimestamp || !this.resultBuffer) {
      return;
    }

    // Only map if buffer is unmapped
    if (this.resultBuffer.mapState === "unmapped") {
      try {
        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());
        // Convert from nanoseconds to microseconds and store as number
        this.lastGpuTime = Number(times[1] - times[0]) / 1000;
        this.resultBuffer.unmap();
      } catch (error) {
        console.warn("Failed to read GPU timing results:", error);
      }
    }
  }

  /**
   * Get the last recorded GPU time in microseconds
   */
  getLastGpuTime(): number {
    return this.lastGpuTime;
  }

  /**
   * Check if GPU timing is supported
   */
  isSupported(): boolean {
    return this.canTimestamp;
  }

  /**
   * Destroy GPU timer resources
   */
  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.resultBuffer?.destroy();
    this.querySet = null;
    this.resolveBuffer = null;
    this.resultBuffer = null;
  }
}
