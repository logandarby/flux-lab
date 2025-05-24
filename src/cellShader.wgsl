
// Vertex Shader code goes here
struct VertexInput {
  @location(0) pos: vec2f,
  @builtin(instance_index) instance: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) cell: vec2f,
};

// Group bindings
@group(0) @binding(0) var<uniform> gridSize: vec2f;
@group(0) @binding(1) var<storage> cellState: array<u32>;

@vertex 
fn vertexMain(input: VertexInput) -> VertexOutput {
  let i = f32(input.instance);
  let cell = vec2f(i % gridSize.x, floor(i / gridSize.x));
  let state = f32(cellState[input.instance]);

  let cellOffset = cell / gridSize * 2;
  // Multiplying by a state of 0 collapses into a single point, which GPU discards
  let gridPos = (input.pos * state + 1) / gridSize - 1 + cellOffset;

  var output: VertexOutput;
  output.pos = vec4f(gridPos, 0, 1);
  output.cell = cell;
  return output;
}

// Fragment Shader code goes here 
// The @location parameter indicates which colorAttachment to draw to
@fragment 
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let c = input.cell / gridSize;
  return vec4f(c, 1 - c.x, 1);
}