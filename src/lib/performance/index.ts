/**
 * Performance Feature Public API
 * Features several utils to measure the performance of several aspects of the application
 *  - Event tracking
 *  - Event performance metrics
 *  - JS Compute time tracking
 *  - WebGPU Compute time tracking
 */

export {
  PerformanceTracker,
  type PerformanceMetrics,
} from "./core/PerformanceTracker";
export { RollingAverage } from "./core/RollingAverage";
export { PerformanceViewer } from "./components/PerformanceViewer";
export { usePerformanceToggle } from "./hooks/usePerformanceToggle";

// Event tracking utilities
export {
  FunctionCallTracker,
  EventCoalescer,
  Throttle,
} from "./utils/eventTracking";

// Event optimization utilities
export { RAFEventProcessor } from "./utils/eventOptimization";

// Event performance types
export type {
  EventPerformanceMetrics,
  ExtendedPerformanceMetrics,
} from "./types/eventMetrics";
