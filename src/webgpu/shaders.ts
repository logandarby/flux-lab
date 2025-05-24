import computeShaderTemplate from '../computeShader.wgsl?raw';

export const createSimulationShaderCode = (workgroupSize: number): string => {
  return computeShaderTemplate.replace(/\$\{WORKGROUP_SIZE\}/g, workgroupSize.toString());
};

export const SQUARE_VERTICES = new Float32Array([
  -0.8, -0.8,
  0.8, -0.8,
  0.8, 0.8,

  -0.8, -0.8,
  0.8, 0.8,
  -0.8, 0.8,
]);

export const VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 8,
  attributes: [{
    format: "float32x2",
    offset: 0,
    shaderLocation: 0,
  }]
}; 