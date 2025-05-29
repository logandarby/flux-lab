struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texCoord: vec2f,
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
    let squareVertices: array<vec2f, 6> = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0),
        vec2f(1.0, -1.0),
        vec2f(1.0, 1.0),
    );
    var output: VertexOutput;
    let xy = squareVertices[vertexIndex];
    output.position = vec4f(xy, 0.0, 1.0);
    // Normalize to 0-1, and flip y
    let normalized = xy * 0.5 + 0.5;
    output.texCoord = vec2f(normalized.x, 1.0 - normalized.y);
    return output;
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

@fragment
fn fragment_main(input: VertexOutput) -> @location(0)vec4f {
    return abs(textureSample(texture, textureSampler, input.texCoord));
}