/**
 * Performance measurement utility for tracking function call frequency
 */
export class FunctionCallTracker {
  private callTimes: number[] = [];
  private lastCleanupTime = performance.now();
  private readonly maxAge: number;
  private readonly name: string;

  constructor(name: string, maxAge = 1000) {
    this.name = name;
    this.maxAge = maxAge; // Keep calls from last N milliseconds
  }

  /**
   * Record a function call
   */
  recordCall(): void {
    const now = performance.now();
    this.callTimes.push(now);

    // Cleanup old entries periodically to prevent memory leaks
    if (now - this.lastCleanupTime > this.maxAge) {
      this.cleanup(now);
      this.lastCleanupTime = now;
    }
  }

  /**
   * Get current calls per second
   */
  getCallsPerSecond(): number {
    const now = performance.now();
    this.cleanup(now);

    if (this.callTimes.length === 0) return 0;

    const windowStart = now - this.maxAge;
    const validCalls = this.callTimes.filter((time) => time > windowStart);

    return validCalls.length * (1000 / this.maxAge);
  }

  /**
   * Get statistics
   */
  getStats(): { name: string; callsPerSecond: number; totalCalls: number } {
    return {
      name: this.name,
      callsPerSecond: this.getCallsPerSecond(),
      totalCalls: this.callTimes.length,
    };
  }

  /**
   * Reset all tracked calls
   */
  reset(): void {
    this.callTimes = [];
    this.lastCleanupTime = performance.now();
  }

  private cleanup(now: number): void {
    const cutoff = now - this.maxAge;
    this.callTimes = this.callTimes.filter((time) => time > cutoff);
  }
}

/**
 * Event coalescing utility for batching rapid events
 */
export class EventCoalescer<T> {
  private pendingEvents: T[] = [];
  private rafId: number | null = null;
  private callback: (events: T[]) => void;

  constructor(callback: (events: T[]) => void) {
    this.callback = callback;
  }

  /**
   * Add an event to be processed
   */
  addEvent(event: T): void {
    this.pendingEvents.push(event);

    if (this.rafId === null) {
      this.rafId = requestAnimationFrame(() => {
        this.flush();
      });
    }
  }

  /**
   * Force flush all pending events
   */
  flush(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }

    if (this.pendingEvents.length > 0) {
      const events = [...this.pendingEvents];
      this.pendingEvents = [];
      this.callback(events);
    }
  }

  /**
   * Get number of pending events
   */
  getPendingCount(): number {
    return this.pendingEvents.length;
  }

  /**
   * Destroy the coalescer
   */
  destroy(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.pendingEvents = [];
  }
}

/**
 * Throttle utility that limits function execution rate
 */
export class Throttle<T extends unknown[] = unknown[]> {
  private lastCallTime = 0;
  private timeoutId: number | null = null;

  constructor(
    private callback: (...args: T) => void,
    private delay: number
  ) {}

  /**
   * Execute the throttled function
   */
  call(...args: T): void {
    const now = performance.now();
    const timeSinceLastCall = now - this.lastCallTime;

    if (timeSinceLastCall >= this.delay) {
      // Execute immediately
      this.lastCallTime = now;
      this.callback(...args);
    } else {
      // Schedule for later if not already scheduled
      if (this.timeoutId === null) {
        const remainingDelay = this.delay - timeSinceLastCall;
        this.timeoutId = window.setTimeout(() => {
          this.timeoutId = null;
          this.lastCallTime = performance.now();
          this.callback(...args);
        }, remainingDelay);
      }
    }
  }

  /**
   * Cancel any pending execution
   */
  cancel(): void {
    if (this.timeoutId !== null) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }

  /**
   * Destroy the throttle
   */
  destroy(): void {
    this.cancel();
  }
}
