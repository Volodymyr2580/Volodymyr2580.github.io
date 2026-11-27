const EPSILON = 1e-7;
const SNAP_DISTANCE = 80;
const point = (value) => ({ x: value.x, y: value.y });
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const finitePoint = (value) => value && Number.isFinite(value.x) && Number.isFinite(value.y);
const cross = (ax, ay, bx, by) => ax * by - ay * bx;

function contains(polygon, x, y) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [ax, ay] = polygon[j];
    const [bx, by] = polygon[i];
    const length = Math.hypot(bx - ax, by - ay);
    if (Math.abs(cross(bx - ax, by - ay, x - ax, y - ay)) <= EPSILON * Math.max(1, length)
      && x >= Math.min(ax, bx) - EPSILON && x <= Math.max(ax, bx) + EPSILON
      && y >= Math.min(ay, by) - EPSILON && y <= Math.max(ay, by) + EPSILON) return true;
    if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) inside = !inside;
  }
  return inside;
}

// Record exact boundary crossings; sampling alone can miss narrow walls or gaps.
function crossings(a, b, polygon) {
  const result = [];
  const rx = b.x - a.x;
  const ry = b.y - a.y;
  const lengthSquared = rx * rx + ry * ry;
  if (lengthSquared === 0) return result;
  for (let i = 0; i < polygon.length; i++) {
    const c = polygon[i];
    const d = polygon[(i + 1) % polygon.length];
    const sx = d[0] - c[0];
    const sy = d[1] - c[1];
    const qx = c[0] - a.x;
    const qy = c[1] - a.y;
    const denominator = cross(rx, ry, sx, sy);
    if (Math.abs(denominator) > EPSILON) {
      const t = cross(qx, qy, sx, sy) / denominator;
      const u = cross(qx, qy, rx, ry) / denominator;
      if (t >= -EPSILON && t <= 1 + EPSILON && u >= -EPSILON && u <= 1 + EPSILON) {
        result.push(Math.max(0, Math.min(1, t)));
      }
    } else if (Math.abs(cross(qx, qy, rx, ry)) <= EPSILON) {
      const t0 = (qx * rx + qy * ry) / lengthSquared;
      const t1 = t0 + (sx * rx + sy * ry) / lengthSquared;
      const lo = Math.max(0, Math.min(t0, t1));
      const hi = Math.min(1, Math.max(t0, t1));
      if (lo <= hi) result.push(lo, hi);
    }
  }
  return result;
}

class MinHeap {
  values = [];
  push(value) {
    const values = this.values;
    let index = values.length;
    values.push(value);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (values[parent].priority <= value.priority) break;
      values[index] = values[parent];
      index = parent;
    }
    values[index] = value;
  }
  pop() {
    const values = this.values;
    const first = values[0];
    const last = values.pop();
    if (values.length) {
      let index = 0;
      while (index * 2 + 1 < values.length) {
        let child = index * 2 + 1;
        if (child + 1 < values.length && values[child + 1].priority < values[child].priority) child++;
        if (values[child].priority >= last.priority) break;
        values[index] = values[child];
        index = child;
      }
      values[index] = last;
    }
    return first;
  }
}

/** Static point navigation in world coordinates. Polygon edges are solid for obstacles. */
export function createNavigation({ width, height, walkablePolygons, obstacles = [], cellSize = 12 }) {
  if (![width, height, cellSize].every((value) => Number.isFinite(value) && value > 0)) {
    throw new RangeError('World dimensions and cellSize must be finite positive numbers.');
  }
  const validPolygon = (polygon) => Array.isArray(polygon) && polygon.length >= 3
    && polygon.every((vertex) => Array.isArray(vertex) && vertex.length === 2 && vertex.every(Number.isFinite));
  if (!Array.isArray(walkablePolygons) || !walkablePolygons.every(validPolygon)
    || !Array.isArray(obstacles) || !obstacles.every(validPolygon)) {
    throw new TypeError('Navigation polygons need at least three finite [x, y] vertices.');
  }
  // Own a snapshot so the precomputed grid cannot drift from collision geometry.
  const terrain = walkablePolygons.map((polygon) => polygon.map((vertex) => [...vertex]));
  const walls = obstacles.map((polygon) => polygon.map((vertex) => [...vertex]));
  const isWalkable = (x, y) => Number.isFinite(x) && Number.isFinite(y)
    && x >= 0 && y >= 0 && x <= width && y <= height
    && terrain.some((polygon) => contains(polygon, x, y))
    && !walls.some((polygon) => contains(polygon, x, y));

  function clearSegment(a, b) {
    if (!isWalkable(a.x, a.y) || !isWalkable(b.x, b.y)) return false;
    if (walls.some((polygon) => crossings(a, b, polygon).length > 0)) return false;
    const cuts = [0, 1, ...terrain.flatMap((polygon) => crossings(a, b, polygon))].sort((a, b) => a - b);
    for (let i = 1; i < cuts.length; i++) {
      const t = (cuts[i] + cuts[i - 1]) / 2;
      if (!isWalkable(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)) return false;
    }
    return true;
  }

  const columns = Math.ceil(width / cellSize);
  const rows = Math.ceil(height / cellSize);
  const nodes = Array.from({ length: columns * rows }, (_, index) => ({
    x: Math.min(width, (index % columns + 0.5) * cellSize),
    y: Math.min(height, (Math.floor(index / columns) + 0.5) * cellSize),
  }));
  const openCells = nodes.map((node) => isWalkable(node.x, node.y));
  const edges = nodes.map(() => null);

  function neighbors(index) {
    if (edges[index]) return edges[index];
    const result = [];
    const x = index % columns;
    const y = Math.floor(index / columns);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if ((!dx && !dy) || nx < 0 || ny < 0 || nx >= columns || ny >= rows) continue;
        const next = ny * columns + nx;
        if (!openCells[next]) continue;
        // Both orthogonal cells must be open before taking a diagonal step.
        if (dx && dy && (!openCells[y * columns + nx] || !openCells[ny * columns + x])) continue;
        if (clearSegment(nodes[index], nodes[next])) result.push(next);
      }
    }
    edges[index] = result;
    return result;
  }

  function reachableToward(a, b) {
    if (clearSegment(a, b)) return point(b);
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 30; i++) {
      const t = (lo + hi) / 2;
      if (clearSegment(a, { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })) lo = t;
      else hi = t;
    }
    return { x: a.x + (b.x - a.x) * lo, y: a.y + (b.y - a.y) * lo };
  }

  function findPath(start, goal) {
    if (!finitePoint(start) || !finitePoint(goal) || !isWalkable(start.x, start.y)) return [];
    if (clearSegment(start, goal)) return distance(start, goal) < EPSILON ? [point(start)] : [point(start), point(goal)];
    const queue = new MinHeap();
    const cost = new Float64Array(nodes.length).fill(Infinity);
    const parent = new Int32Array(nodes.length).fill(-1);
    const visited = new Uint8Array(nodes.length);
    // Multiple local anchors avoid making the nearest grid cell a bottleneck.
    const anchorRadius = cellSize * 1.6;
    for (let index = 0; index < nodes.length; index++) {
      const d = distance(start, nodes[index]);
      if (openCells[index] && d <= anchorRadius && clearSegment(start, nodes[index])) {
        cost[index] = d;
        queue.push({ index, priority: d + distance(nodes[index], goal) });
      }
    }
    let bestIndex = -1;
    let bestDistance = Infinity;
    let finalPoint = null;
    while (queue.values.length) {
      const { index } = queue.pop();
      if (visited[index]) continue;
      visited[index] = 1;
      const d = distance(nodes[index], goal);
      if (d < bestDistance) {
        bestDistance = d;
        bestIndex = index;
      }
      if (d <= anchorRadius && clearSegment(nodes[index], goal)) {
        bestIndex = index;
        finalPoint = point(goal);
        break;
      }
      for (const next of neighbors(index)) {
        if (visited[next]) continue;
        const candidate = cost[index] + distance(nodes[index], nodes[next]);
        if (candidate >= cost[next]) continue;
        cost[next] = candidate;
        parent[next] = index;
        queue.push({ index: next, priority: candidate + distance(nodes[next], goal) });
      }
    }
    if (bestIndex < 0) return [];
    finalPoint ??= reachableToward(nodes[bestIndex], goal);
    if (distance(finalPoint, goal) > SNAP_DISTANCE) return [];
    const route = [];
    for (let index = bestIndex; index >= 0; index = parent[index]) route.push(point(nodes[index]));
    route.reverse();
    route.unshift(point(start));
    route.push(finalPoint);
    // Keep necessary turns while removing visible grid zigzags.
    const smoothed = [route[0]];
    for (let current = 0; current < route.length - 1;) {
      let next = route.length - 1;
      while (next > current + 1 && !clearSegment(route[current], route[next])) next--;
      if (distance(smoothed.at(-1), route[next]) > EPSILON) smoothed.push(route[next]);
      current = next;
    }
    return smoothed;
  }

  function move(position, delta) {
    const result = point(position);
    if (!finitePoint(position) || !finitePoint(delta) || !isWalkable(position.x, position.y)) return result;
    const steps = Math.min(4096, Math.max(1, Math.ceil(Math.hypot(delta.x, delta.y) / Math.min(4, cellSize / 3))));
    const dx = delta.x / steps;
    const dy = delta.y / steps;
    for (let step = 0; step < steps; step++) {
      const next = { x: result.x + dx, y: result.y + dy };
      if (clearSegment(result, next)) Object.assign(result, next);
      else {
        // Resolve each axis separately, advancing up to a wall to avoid sticky edges.
        if (dx) Object.assign(result, reachableToward(result, { x: result.x + dx, y: result.y }));
        if (dy) Object.assign(result, reachableToward(result, { x: result.x, y: result.y + dy }));
      }
    }
    return result;
  }

  return { isWalkable, findPath, move };
}
