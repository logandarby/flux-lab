export { default as SmokeSimulation } from "./core/SmokeSimulation";
export * from "./core/SimulationPasses";
export * from "./core/constants";

// Component exports
export { default as SimulationControls } from "./components/SimulationControls";

// Test component exports (for development)
export { default as AdvectionTestComponent } from "./components/tests/AdvectionTestComponent";
export { default as DiffusionTestComponent } from "./components/tests/DiffusionTestComponent";
export { default as BaseTestComponent } from "./components/tests/BaseTestComponent";
