// --- VERTEX --- //

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

// --- FRAGMENT --- //

struct FragmentInput {
    shaderMode: u32
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;
@group(0) @binding(2) var<uniform> uniformInput: FragmentInput;

// All components are in the range [0…1], including hue.
fn hsv2rgb(c: vec3f) -> vec3f
{
    let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

fn shadeVelocity(value: vec2f, input: VertexOutput) -> vec4f {
    // Velocity
    let pi = 3.14159;
    // let angle = atan(value.y / value.x);
    let angle = acos((value.y * value.y) / (dot(value.xy, value.xy)));
    let magnitude = sqrt(dot(value.xy, value.xy));
    // let color = vec3f(cos(angle), cos(angle + 2 * pi / 3), cos(angle - 2 * pi / 3));
    let color = hsv2rgb(vec3f(angle / 8 + pi / 2, 1, 1));
    return vec4f(color * magnitude, 0);
}

fn shadePressure(value: f32, input: VertexOutput) -> vec4f {
    // Pressure
    if (value < 0) {
        return vec4f(-value, 0, 0, 1);
    } else {
        return vec4f(0, 0, value, 1);
    }
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0)vec4f {
    let DAMP: f32 = 2;
    let value = textureSample(texture, textureSampler, input.texCoord) / DAMP;

    switch uniformInput.shaderMode {
        case 0: {
            return shadeVelocity(value.xy, input);
        }
        case 1: {
            return shadePressure(value.x, input);
        }
        default: {
            return vec4(1, 1, 1, 1);
        }
    }
}