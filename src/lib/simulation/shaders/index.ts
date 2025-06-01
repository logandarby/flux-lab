import { shaderRegistry } from "@/lib/preprocessor";

import advectionShader from "./advectionShader.wgsl?raw";
import jacobiIteration from "./jacobiIteration.wgsl?raw";
import divergenceShader from "./divergenceShader.wgsl?raw";
import gradientSubtractionShader from "./gradientSubtractionShader.wgsl?raw";
import boundaryConditionsShader from "./boundaryConditionsShader.wgsl?raw";
import addSmokeShader from "./addSmokeShader.wgsl?raw";
import addVelocityShader from "./addVelocityShader.wgsl?raw";
import dissipationShader from "./dissipationShader.wgsl?raw";
import bilinearInterpolate from "./bilinearInterpolate.wgsl?raw";
import textureShader from "./textureShader.wgsl?raw";

/**
 * Available shader names with type safety
 * These are the names of the shaders that will be used in the includes names
 */
export const SHADERS = {
  ADVECTION: "advectionShader.wgsl",
  JACOBI_ITERATION: "jacobiIteration.wgsl",
  DIVERGENCE: "divergenceShader.wgsl",
  GRADIENT_SUBTRACTION: "gradientSubtractionShader.wgsl",
  BOUNDARY_CONDITIONS: "boundaryConditionsShader.wgsl",
  ADD_SMOKE: "addSmokeShader.wgsl",
  ADD_VELOCITY: "addVelocityShader.wgsl",
  DISSIPATION: "dissipationShader.wgsl",
  BILINEAR_INTERPOLATE: "bilinearInterpolate.wgsl",
  TEXTURE: "textureShader.wgsl",
} as const;

export type ShaderName = (typeof SHADERS)[keyof typeof SHADERS];

// Auto-register all shaders when this module is imported
shaderRegistry.registerMultiple({
  [SHADERS.ADVECTION]: advectionShader,
  [SHADERS.JACOBI_ITERATION]: jacobiIteration,
  [SHADERS.DIVERGENCE]: divergenceShader,
  [SHADERS.GRADIENT_SUBTRACTION]: gradientSubtractionShader,
  [SHADERS.BOUNDARY_CONDITIONS]: boundaryConditionsShader,
  [SHADERS.ADD_SMOKE]: addSmokeShader,
  [SHADERS.ADD_VELOCITY]: addVelocityShader,
  [SHADERS.DISSIPATION]: dissipationShader,
  [SHADERS.BILINEAR_INTERPOLATE]: bilinearInterpolate,
  [SHADERS.TEXTURE]: textureShader,
});

/**
 * Type-safe wrapper for wgsl function with simulation shaders
 */
export { wgsl } from "@/lib/preprocessor";
