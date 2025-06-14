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

  // Lava lamp mode configuration
  lavaLamp: {
    spawnAreaHeightPercent: number; // Percentage of screen height from top where pills spawn
    velocityRange: {
      horizontal: { min: number; max: number }; // Horizontal velocity range
      vertical: { min: number; max: number }; // Vertical velocity range (positive = downward)
    };
    pillDuration: { min: number; max: number }; // Duration each pill lasts (ms)
    pillInterval: number; // How often smoke/velocity is added during pill (ms)
    spawnInterval: { min: number; max: number }; // Time between new pills (ms)
    velocityScale: number; // Scale factor for pill movement

    // Sub-pill trail configuration
    subPills: {
      count: { min: number; max: number }; // Number of sub-pills per main pill
      spawnDelay: { min: number; max: number }; // Delay between sub-pill spawns (ms)
      velocityVariation: number; // Percentage of main velocity variation (0.0 to 1.0)
      positionVariation: number; // Pixel variation in spawn position
    };
  };
}
