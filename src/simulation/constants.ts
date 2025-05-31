import type { ShaderMode } from "@/utils/RenderPass";

// Configuration types for persistent settings
export interface SmokeSimulationConfig {
  shaderMode: ShaderMode;
  texture: SmokeTextureID;
}

// Velocity simulation step controls
export interface VelocitySimulationControls {
  enableAdvection?: boolean;
  enableDiffusion?: boolean;
  enableDivergence?: boolean;
  enablePressureProjection?: boolean;
  enablePressureBoundaryConditions?: boolean;
  enableGradientSubtraction?: boolean;
  enableVelocityBoundaryConditions?: boolean;
  enableDissipation?: boolean;
}

// Smoke simulation step controls
export interface SmokeSimulationControls {
  enableAdvection?: boolean;
  enableBoundaryConditions?: boolean;
  enableDiffusion?: boolean;
  enableDissipation?: boolean;
}

// Overall step configuration
export interface SimulationStepConfig {
  renderOnly?: boolean;
  shaderMode?: ShaderMode;
  texture?: SmokeTextureID;
  velocity?: VelocitySimulationControls;
  smoke?: SmokeSimulationControls;
}

// Default configurations
export const DEFAULT_VELOCITY_CONTROLS: Required<VelocitySimulationControls> = {
  enableAdvection: true,
  enableDiffusion: true,
  enableDivergence: true,
  enablePressureProjection: true,
  enablePressureBoundaryConditions: true,
  enableGradientSubtraction: true,
  enableVelocityBoundaryConditions: true,
  enableDissipation: true,
};

export const DEFAULT_SMOKE_CONTROLS: Required<SmokeSimulationControls> = {
  enableAdvection: true,
  enableBoundaryConditions: true,
  enableDiffusion: true,
  enableDissipation: true,
};

// Simulation constants interface
export interface SimulationConstants {
  // Grid configuration
  grid: {
    size: { width: number; height: number };
    scale: number;
  };

  // Compute configuration
  compute: {
    workgroupSize: number;
    workgroupCount: number;
  };

  // Physics parameters
  physics: {
    timestep: number;
    diffusionFactor: number;
    velocityAdvection: number;
    smokeAdvection: number;
    smokeDiffusion: number;
    smokeDissipationFactor: number;
    velocityDissipationFactor: number;
  };

  // Interaction parameters
  interaction: {
    radius: number;
    smokeIntensity: number;
    velocityIntensity: number;
  };

  // Iteration counts
  iterations: {
    diffusion: number;
    pressure: number;
  };

  // Particle configuration
  particles: {
    smokeDimensions: { width: number; height: number };
  };
}

// Default simulation constants
export const SIMULATION_CONSTANTS: SimulationConstants = {
  grid: {
    size: { width: 2 ** 9, height: 2 ** 9 },
    scale: 1,
  },

  compute: {
    workgroupSize: 16,
    get workgroupCount() {
      return Math.ceil(
        SIMULATION_CONSTANTS.grid.size.width / this.workgroupSize
      );
    },
  },

  physics: {
    timestep: 1.0 / 60.0,
    diffusionFactor: 1, // Smaller = slower diffusion
    velocityAdvection: 10, // Smaller = slower velocity advection
    smokeAdvection: 20,
    smokeDiffusion: 1,
    smokeDissipationFactor: 0.99, // Multiplication factor for smoke density each frame
    velocityDissipationFactor: 0.995, // Similarly for velocity magnitude
  },

  interaction: {
    radius: 20, // Radius of effect when adding smoke/velocity
    smokeIntensity: 0.1, // Intensity of smoke when clicking
    velocityIntensity: 2.0, // Intensity multiplier for velocity when dragging
  },

  iterations: {
    diffusion: 40,
    pressure: 100,
  },

  particles: {
    smokeDimensions: { width: 2 ** 9, height: 2 ** 9 }, // Match the grid size
  },
};

// Utils

export type SmokeTextureID =
  | "divergence"
  | "velocity"
  | "pressure"
  | "smokeDensity"
  | "smokeParticlePosition";
