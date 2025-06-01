export class RollingAverage {
  private samples: number[] = [];
  private maxSamples: number;

  constructor(maxSamples = 60) {
    this.maxSamples = maxSamples;
  }

  addSample(value: number): void {
    if (value < 0) return;

    this.samples.push(value);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
  }

  get(): number {
    if (this.samples.length === 0) return 0;
    const sum = this.samples.reduce((acc, val) => acc + val, 0);
    return sum / this.samples.length;
  }

  clear(): void {
    this.samples = [];
  }
}
