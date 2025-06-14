import type SmokeSimulation from "@/lib/simulation/core/SmokeSimulation";

/**
 * Canvas bounds cache for performance optimization
 */
export class CanvasBoundsCache {
  private bounds: DOMRect | null = null;
  private updateTime = 0;
  private readonly cacheTimeout: number;

  constructor(cacheTimeoutMs: number = 100) {
    this.cacheTimeout = cacheTimeoutMs;
  }

  /**
   * Get cached canvas bounds, updating if necessary
   */
  getBounds(canvas: HTMLCanvasElement | null): DOMRect | null {
    const now = performance.now();

    if (!this.bounds || !canvas || now - this.updateTime > this.cacheTimeout) {
      if (canvas) {
        this.bounds = canvas.getBoundingClientRect();
        this.updateTime = now;
      }
    }

    return this.bounds;
  }

  /**
   * Clear the cached bounds (useful when canvas resizes)
   */
  clear(): void {
    this.bounds = null;
  }
}

/**
 * Convert client coordinates to grid coordinates
 */
export function clientToGridCoords(
  clientX: number,
  clientY: number,
  canvasBounds: DOMRect,
  gridSize: { width: number; height: number }
): { x: number; y: number } | null {
  const canvasX = clientX - canvasBounds.left;
  const canvasY = clientY - canvasBounds.top;

  // Scale to grid coordinates
  const gridX = (canvasX / canvasBounds.width) * gridSize.width;
  // Note: Y coordinate should NOT be flipped since both canvas and WebGPU use top-left origin
  const gridY = (canvasY / canvasBounds.height) * gridSize.height;

  return { x: gridX, y: gridY };
}

/**
 * Convert grid coordinates to client coordinates
 */
export function gridToClientCoords(
  gridX: number,
  gridY: number,
  canvasBounds: DOMRect,
  gridSize: { width: number; height: number }
): { clientX: number; clientY: number } {
  // Convert grid coordinates to canvas coordinates
  const canvasX = (gridX / gridSize.width) * canvasBounds.width;
  const canvasY = (gridY / gridSize.height) * canvasBounds.height;

  // Convert to client coordinates
  const clientX = canvasX + canvasBounds.left;
  const clientY = canvasY + canvasBounds.top;

  return { clientX, clientY };
}

/**
 * Canvas coordinate conversion utilities for simulation components
 */
export class CanvasCoordinateConverter {
  private boundsCache: CanvasBoundsCache;
  private canvas: HTMLCanvasElement | null = null;
  private simulation: SmokeSimulation | null = null;

  constructor(cacheTimeoutMs: number = 100) {
    this.boundsCache = new CanvasBoundsCache(cacheTimeoutMs);
  }

  /**
   * Initialize with canvas and simulation references
   */
  initialize(canvas: HTMLCanvasElement, simulation: SmokeSimulation): void {
    this.canvas = canvas;
    this.simulation = simulation;
  }

  /**
   * Clear canvas bounds cache (call on resize)
   */
  clearCache(): void {
    this.boundsCache.clear();
  }

  /**
   * Convert client coordinates to grid coordinates
   */
  clientToGrid(
    clientX: number,
    clientY: number
  ): { x: number; y: number } | null {
    if (!this.canvas || !this.simulation) return null;

    const bounds = this.boundsCache.getBounds(this.canvas);
    const gridSize = this.simulation.getGridSize();

    if (!bounds || !gridSize) return null;

    return clientToGridCoords(clientX, clientY, bounds, gridSize);
  }

  /**
   * Convert grid coordinates to client coordinates
   */
  gridToClient(
    gridX: number,
    gridY: number
  ): { clientX: number; clientY: number } | null {
    if (!this.canvas || !this.simulation) return null;

    const bounds = this.boundsCache.getBounds(this.canvas);
    const gridSize = this.simulation.getGridSize();

    if (!bounds || !gridSize) return null;

    return gridToClientCoords(gridX, gridY, bounds, gridSize);
  }
}

/**
 * Utility for handling mouse activity detection
 */
export function isMouseNearBottom(
  clientY: number,
  threshold: number = 150
): boolean {
  const distanceFromBottom = window.innerHeight - clientY;
  return distanceFromBottom <= threshold;
}
