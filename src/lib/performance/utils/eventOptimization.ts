/**
 * RAF-based event processor that batches discrete events until the next animation frame
 *
 * Used for handling events that are not tied to a specific animation frame, such as mouse events and keyboard events
 */
export class RAFEventProcessor<T> {
  private pendingEvents: T[] = [];
  private rafId: number | null = null;
  private isProcessing = false;
  private callback: (events: T[]) => void;
  private onEventBatch?: (batchSize: number) => void;

  constructor(
    callback: (events: T[]) => void,
    onEventBatch?: (batchSize: number) => void
  ) {
    this.callback = callback;
    this.onEventBatch = onEventBatch;
  }

  /**
   * Add an event to be processed on the next animation frame
   */
  addEvent(event: T): void {
    this.pendingEvents.push(event);

    if (this.rafId === null && !this.isProcessing) {
      this.rafId = requestAnimationFrame(() => {
        this.processBatch();
      });
    }
  }

  /**
   * Destroy the processor and clean up
   */
  destroy(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.pendingEvents = [];
  }

  private processBatch(): void {
    if (this.pendingEvents.length === 0) return;

    this.isProcessing = true;
    const events = [...this.pendingEvents];
    this.pendingEvents = [];
    this.rafId = null;

    // Report batch size for analytics
    if (this.onEventBatch) {
      this.onEventBatch(events.length);
    }

    try {
      this.callback(events);
    } finally {
      this.isProcessing = false;
    }
  }
}
