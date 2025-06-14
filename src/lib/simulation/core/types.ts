import type { ShaderMode } from "@/shared/webgpu/RenderPass";

export type SmokeTextureID =
  | "divergence"
  | "velocity"
  | "pressure"
  | "smokeDensity"
  | "smokeParticlePosition";

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
  enableAdvectParticles?: boolean;
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
  enableAdvectParticles: true,
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
    maxTimestep: number;
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

  // Noise configuration
  noise: {
    stddev: number;
    mean: number;
  };
}
