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
    let DAMP: f32 = 2;
    let value = textureSample(texture, textureSampler, input.texCoord) / DAMP;

    // Pressure
    // if (value.x < 0) {
    //     return vec4f(-value.x, 0, 0, 1);
    // } else {
    //     return vec4f(0, 0, value.x, 1);
    // }

    // Velocity
    let pi = 3.14159;
    // let angle = atan(value.y / value.x);
    let angle = acos((value.y * value.y) / (dot(value.xy, value.xy)));
    let magnitude = sqrt(dot(value.xy, value.xy));
    let color = vec3f(cos(angle), cos(angle + 2 * pi / 3), cos(angle - 2 * pi / 3));
    return vec4f(color * magnitude, 0);
}