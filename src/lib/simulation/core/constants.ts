import type { SimulationConstants } from "./types";

export const FLOAT_BYTES = 4;

// Default simulation constants
export const SIMULATION_CONSTANTS: SimulationConstants = {
  grid: {
    size: { width: 2 ** 9, height: 2 ** 8 },
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
    maxTimestep: 1.0 / 30.0,
    diffusionFactor: 10, // Smaller = slower diffusion
    velocityAdvection: 10, // Smaller = slower velocity advection
    smokeAdvection: 20,
    smokeDiffusion: 1.1,
    smokeDissipationFactor: 0.995, // Multiplication factor for smoke density each frame
    velocityDissipationFactor: 0.995, // Similarly for velocity magnitude
  },

  interaction: {
    radius: 20, // Radius of effect when adding smoke/velocity
    smokeIntensity: 0.18, // Intensity of smoke when clicking
    velocityIntensity: 2.0, // Intensity multiplier for velocity when dragging
  },

  iterations: {
    diffusion: 25,
    pressure: 80,
  },

  particles: {
    smokeDimensions: { width: 2 ** 7, height: 2 ** 7 },
  },

  noise: {
    stddev: 0.06,
    mean: 0.0,
  },
};
