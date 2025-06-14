import type SmokeSimulation from "@/lib/simulation/core/SmokeSimulation";
import type { useAudioVisualization } from "@/lib/audio-visualization";
import { SIMULATION_CONSTANTS } from "../core/constants";
import { gridToClientCoords } from "./canvasUtils";

/**
 * Configuration for creating a lava lamp sub-pill
 */
export interface SubPillConfig {
  baseX: number;
  baseY: number;
  baseVelocityX: number;
  baseVelocityY: number;
  baseDuration: number;
  lavaLampConfig: typeof SIMULATION_CONSTANTS.lavaLamp;
  gridSize: { width: number; height: number };
}

/**
 * Dependencies needed for lava lamp functionality
 */
export interface LavaLampDependencies {
  simulation: SmokeSimulation;
  audioVisualization: ReturnType<typeof useAudioVisualization>;
  canvasBounds: DOMRect;
  isLavaLampMode: boolean;
  intervalTracker: Set<number>;
}

/**
 * Create a single sub-pill with variations from the main pill parameters
 */
export function createSubPill(
  config: SubPillConfig,
  dependencies: LavaLampDependencies
): void {
  const {
    baseX,
    baseY,
    baseVelocityX,
    baseVelocityY,
    baseDuration,
    lavaLampConfig,
    gridSize,
  } = config;

  const { simulation, canvasBounds, isLavaLampMode, intervalTracker } =
    dependencies;

  // Apply position variation
  const x =
    baseX +
    (Math.random() - 0.5) * 2 * lavaLampConfig.subPills.positionVariation;
  const y =
    baseY +
    (Math.random() - 0.5) * 2 * lavaLampConfig.subPills.positionVariation;

  // Apply velocity variation (much smaller than main pill)
  const velocityVariation = lavaLampConfig.subPills.velocityVariation;
  const velocityX =
    baseVelocityX +
    (Math.random() - 0.5) * 2 * Math.abs(baseVelocityX) * velocityVariation;
  const velocityY =
    baseVelocityY +
    (Math.random() - 0.5) * 2 * Math.abs(baseVelocityY) * velocityVariation;

  // Slight duration variation (±20%)
  const pillDuration =
    baseDuration + (Math.random() - 0.5) * baseDuration * 0.4;
  const pillInterval = lavaLampConfig.pillInterval;

  let pillTimeElapsed = 0;
  let pillStarted = false;

  const pillIntervalId = setInterval(() => {
    if (!simulation || !isLavaLampMode) {
      // Cleanup: send mouse up event if pill was active
      if (pillStarted) {
        const currentX =
          x +
          ((velocityX * pillTimeElapsed) / 1000) * lavaLampConfig.velocityScale;
        const currentY =
          y +
          ((velocityY * pillTimeElapsed) / 1000) * lavaLampConfig.velocityScale;
        const clientCoords = gridToClientCoords(
          currentX,
          currentY,
          canvasBounds,
          gridSize
        );
        if (clientCoords) {
          //   audioVisualization.handleMouseUp(clientCoords.clientX, clientCoords.clientY);
        }
      }
      clearInterval(pillIntervalId);
      intervalTracker.delete(pillIntervalId);
      return;
    }

    // Add smoke and velocity at the current position
    const currentX =
      x + ((velocityX * pillTimeElapsed) / 1000) * lavaLampConfig.velocityScale;
    const currentY =
      y + ((velocityY * pillTimeElapsed) / 1000) * lavaLampConfig.velocityScale;

    // Keep within bounds
    if (
      currentX >= 0 &&
      currentX < gridSize.width &&
      currentY >= 0 &&
      currentY < gridSize.height
    ) {
      simulation.addSmoke(currentX, currentY);
      simulation.addVelocity(currentX, currentY, velocityX, velocityY);

      // Send audio events
      const clientCoords = gridToClientCoords(
        currentX,
        currentY,
        canvasBounds,
        gridSize
      );
      if (clientCoords) {
        if (!pillStarted) {
          // First frame - send mouse down
          //   audioVisualization.handleMouseDown(clientCoords.clientX, clientCoords.clientY);
          pillStarted = true;
        } else {
          // Subsequent frames - send mouse move
          //   audioVisualization.handleMouseMove(clientCoords.clientX, clientCoords.clientY);
        }
      }
    }

    pillTimeElapsed += pillInterval;

    // Stop the pill after its duration
    if (pillTimeElapsed >= pillDuration) {
      // Send mouse up event before stopping
      if (pillStarted) {
        const finalX =
          x +
          ((velocityX * pillTimeElapsed) / 1000) * lavaLampConfig.velocityScale;
        const finalY =
          y +
          ((velocityY * pillTimeElapsed) / 1000) * lavaLampConfig.velocityScale;
        const clientCoords = gridToClientCoords(
          finalX,
          finalY,
          canvasBounds,
          gridSize
        );
        if (clientCoords) {
          //   audioVisualization.handleMouseUp(clientCoords.clientX, clientCoords.clientY);
        }
      }
      clearInterval(pillIntervalId);
      intervalTracker.delete(pillIntervalId);
    }
  }, pillInterval);

  intervalTracker.add(pillIntervalId);
}

/**
 * Enum for pill spawn edges
 */
export enum PillSpawnEdge {
  TOP = "top",
  BOTTOM = "bottom",
  LEFT = "left",
  RIGHT = "right",
}

/**
 * Create a lava lamp pill (which spawns multiple sub-pills)
 */
export function createLavaLampPill(dependencies: LavaLampDependencies): void {
  const { simulation, intervalTracker } = dependencies;

  if (!simulation) return;

  const gridSize = simulation.getGridSize();
  if (!gridSize) return;

  const lavaLampConfig = SIMULATION_CONSTANTS.lavaLamp;

  // Randomly choose which edge to spawn from
  const edges = [
    PillSpawnEdge.TOP,
    PillSpawnEdge.BOTTOM,
    PillSpawnEdge.LEFT,
    PillSpawnEdge.RIGHT,
  ];
  const spawnEdge = edges[Math.floor(Math.random() * edges.length)];

  let baseX: number;
  let baseY: number;
  let baseVelocityX: number;
  let baseVelocityY: number;

  // Calculate position and velocity based on spawn edge
  switch (spawnEdge) {
    case PillSpawnEdge.TOP:
      // Spawn at top, move downward
      baseX = Math.random() * gridSize.width;
      baseY =
        Math.random() *
        (gridSize.height * lavaLampConfig.spawnAreaHeightPercent);
      baseVelocityX =
        Math.random() *
          (lavaLampConfig.velocityRange.horizontal.max -
            lavaLampConfig.velocityRange.horizontal.min) +
        lavaLampConfig.velocityRange.horizontal.min;
      baseVelocityY =
        Math.random() *
          (lavaLampConfig.velocityRange.vertical.max -
            lavaLampConfig.velocityRange.vertical.min) +
        lavaLampConfig.velocityRange.vertical.min;
      break;

    case PillSpawnEdge.BOTTOM:
      // Spawn at bottom, move upward
      baseX = Math.random() * gridSize.width;
      baseY =
        gridSize.height -
        Math.random() *
          (gridSize.height * lavaLampConfig.spawnAreaHeightPercent);
      baseVelocityX =
        Math.random() *
          (lavaLampConfig.velocityRange.horizontal.max -
            lavaLampConfig.velocityRange.horizontal.min) +
        lavaLampConfig.velocityRange.horizontal.min;
      // Negative Y velocity to move upward
      baseVelocityY = -(
        Math.random() *
          (lavaLampConfig.velocityRange.vertical.max -
            lavaLampConfig.velocityRange.vertical.min) +
        lavaLampConfig.velocityRange.vertical.min
      );
      break;

    case PillSpawnEdge.LEFT:
      // Spawn at left, move rightward
      baseX =
        Math.random() *
        (gridSize.width * lavaLampConfig.spawnAreaHeightPercent);
      baseY = Math.random() * gridSize.height;
      // Positive X velocity to move rightward
      baseVelocityX =
        Math.random() *
          (lavaLampConfig.velocityRange.vertical.max -
            lavaLampConfig.velocityRange.vertical.min) +
        lavaLampConfig.velocityRange.vertical.min;
      baseVelocityY =
        Math.random() *
          (lavaLampConfig.velocityRange.horizontal.max -
            lavaLampConfig.velocityRange.horizontal.min) +
        lavaLampConfig.velocityRange.horizontal.min;
      break;

    case PillSpawnEdge.RIGHT:
      // Spawn at right, move leftward
      baseX =
        gridSize.width -
        Math.random() *
          (gridSize.width * lavaLampConfig.spawnAreaHeightPercent);
      baseY = Math.random() * gridSize.height;
      // Negative X velocity to move leftward
      baseVelocityX = -(
        Math.random() *
          (lavaLampConfig.velocityRange.vertical.max -
            lavaLampConfig.velocityRange.vertical.min) +
        lavaLampConfig.velocityRange.vertical.min
      );
      baseVelocityY =
        Math.random() *
          (lavaLampConfig.velocityRange.horizontal.max -
            lavaLampConfig.velocityRange.horizontal.min) +
        lavaLampConfig.velocityRange.horizontal.min;
      break;
  }

  // Random base duration
  const baseDuration =
    Math.random() *
      (lavaLampConfig.pillDuration.max - lavaLampConfig.pillDuration.min) +
    lavaLampConfig.pillDuration.min;

  // Random number of sub-pills
  const subPillCount =
    Math.floor(
      Math.random() *
        (lavaLampConfig.subPills.count.max -
          lavaLampConfig.subPills.count.min +
          1)
    ) + lavaLampConfig.subPills.count.min;

  // Create sub-pills with staggered timing
  for (let i = 0; i < subPillCount; i++) {
    const spawnDelay =
      Math.random() *
        (lavaLampConfig.subPills.spawnDelay.max -
          lavaLampConfig.subPills.spawnDelay.min) +
      lavaLampConfig.subPills.spawnDelay.min;

    const timeoutId = setTimeout(() => {
      const subPillConfig: SubPillConfig = {
        baseX,
        baseY,
        baseVelocityX,
        baseVelocityY,
        baseDuration,
        lavaLampConfig,
        gridSize,
      };

      createSubPill(subPillConfig, dependencies);
      intervalTracker.delete(timeoutId);
    }, i * spawnDelay);

    intervalTracker.add(timeoutId);
  }
}

/**
 * Schedule the next lava lamp pill
 */
export function scheduleLavaLampPill(
  dependencies: LavaLampDependencies,
  scheduleNext: () => void
): void {
  const { isLavaLampMode, intervalTracker } = dependencies;

  if (!isLavaLampMode) return;

  const lavaLampConfig = SIMULATION_CONSTANTS.lavaLamp;

  // Random interval between pills
  const nextPillDelay =
    Math.random() *
      (lavaLampConfig.spawnInterval.max - lavaLampConfig.spawnInterval.min) +
    lavaLampConfig.spawnInterval.min;

  const timeoutId = setTimeout(() => {
    createLavaLampPill(dependencies);
    intervalTracker.delete(timeoutId);

    // Schedule the next pill
    scheduleNext();
  }, nextPillDelay);

  intervalTracker.add(timeoutId);
}

/**
 * Clear all lava lamp intervals
 */
export function clearLavaLampIntervals(intervalTracker: Set<number>): void {
  intervalTracker.forEach((intervalId) => {
    clearTimeout(intervalId);
  });
  intervalTracker.clear();
}
