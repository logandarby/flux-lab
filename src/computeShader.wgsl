@group(0) @binding(0) var<uniform> gridSize: vec2f;

// Two buffers for pingpong pattern
@group(0) @binding(1) var<storage> cellStateIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

// Gets the cell index (instance_id) from the 2d position (global_invocation_id)
fn cellIndex(cell: vec2u) -> u32 {
  return (cell.y % u32(gridSize.y)) * u32(gridSize.x) + (cell.x % u32(gridSize.x));
}

fn cellActive(x: u32, y: u32) -> u32 {
  return cellStateIn[cellIndex(vec2(x, y))];
}

@compute
@workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
  let nActiveNeighbours =
    cellActive(cell.x + 1, cell.y + 1) +
    cellActive(cell.x + 1, cell.y + 0) +
    cellActive(cell.x + 1, cell.y - 1) +
    cellActive(cell.x + 0, cell.y + 1) +
    cellActive(cell.x + 0, cell.y + 0) +
    cellActive(cell.x + 0, cell.y - 1) +
    cellActive(cell.x - 1, cell.y + 1) +
    cellActive(cell.x - 1, cell.y + 0) +
    cellActive(cell.x - 1, cell.y - 1);

  let i = cellIndex(cell.xy);
  switch (nActiveNeighbours) {
    case 2: {
      cellStateOut[i] = cellStateIn[i];
    }
    case 3: {
      cellStateOut[i] = 1;
    }
    default: {
      cellStateOut[i] = 0;
    }
  }
} 