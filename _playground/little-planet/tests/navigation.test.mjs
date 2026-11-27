import test from 'node:test';
import assert from 'node:assert/strict';
import { createNavigation } from '../src/navigation.js';
import { WORLD, SPAWN, WALKABLE, PLACEMENTS } from '../src/world-map.js';

const rectangle = (x0, y0, x1, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
const create = (options = {}) => createNavigation({
  width: 240, height: 160, walkablePolygons: [rectangle(0, 0, 240, 160)], cellSize: 10, ...options,
});
const close = (actual, expected, tolerance = 1e-5) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
function assertTraversable(nav, path) {
  assert.ok(path.length > 0);
  for (let i = 0; i < path.length; i++) {
    assert.ok(nav.isWalkable(path[i].x, path[i].y));
    if (!i) continue;
    const target = nav.move(path[i - 1], { x: path[i].x - path[i - 1].x, y: path[i].y - path[i - 1].y });
    close(target.x, path[i].x);
    close(target.y, path[i].y);
  }
}

test('terrain boundaries are walkable, obstacle boundaries and outside world are blocked', () => {
  const nav = create({ obstacles: [rectangle(30, 30, 50, 50)] });
  assert.equal(nav.isWalkable(0, 0), true);
  assert.equal(nav.isWalkable(240, 160), true);
  assert.equal(nav.isWalkable(-0.001, 10), false);
  assert.equal(nav.isWalkable(241, 10), false);
  assert.equal(nav.isWalkable(30, 40), false);
  assert.equal(nav.isWalkable(40, 40), false);
  assert.equal(nav.isWalkable(NaN, 0), false);
});

test('direct paths preserve exact endpoints, including off-grid starts', () => {
  const nav = create();
  const start = { x: 1, y: 3 };
  const goal = { x: 232, y: 151 };
  assert.deepEqual(nav.findPath(start, goal), [start, goal]);
  assert.deepEqual(nav.findPath(start, start), [start]);
});

test('A* finds a traversable route around a long wall', () => {
  const nav = create({ obstacles: [rectangle(105, 0, 125, 115)] });
  const start = { x: 35, y: 30 };
  const goal = { x: 200, y: 30 };
  const path = nav.findPath(start, goal);
  assert.deepEqual(path[0], start);
  assert.deepEqual(path.at(-1), goal);
  assert.ok(path.some(({ y }) => y > 115));
  assertTraversable(nav, path);
});

test('diagonally touching obstacles cannot be squeezed through at their shared corner', () => {
  const nav = create({ width: 300, height: 300, walkablePolygons: [rectangle(0, 0, 300, 300)],
    obstacles: [rectangle(150, 0, 300, 150), rectangle(0, 150, 150, 300)] });
  assert.deepEqual(nav.findPath({ x: 50, y: 50 }, { x: 250, y: 250 }), []);
  const result = nav.move({ x: 100, y: 100 }, { x: 100, y: 100 });
  assert.ok(result.x < 150 && result.y < 150);
});

test('off-terrain clicks snap to reachable terrain only within 80 units', () => {
  const nav = create({ walkablePolygons: [rectangle(10, 10, 100, 150)] });
  const near = nav.findPath({ x: 20, y: 70 }, { x: 145, y: 70 });
  assertTraversable(nav, near);
  close(near.at(-1).x, 100);
  assert.deepEqual(nav.findPath({ x: 20, y: 70 }, { x: 200, y: 70 }), []);
});

test('nearby disconnected islands snap to this island and distant islands return no route', () => {
  const nav = create({ width: 400, walkablePolygons: [rectangle(0, 0, 100, 160), rectangle(140, 0, 400, 160)] });
  const path = nav.findPath({ x: 30, y: 70 }, { x: 150, y: 70 });
  assertTraversable(nav, path);
  assert.ok(path.every(({ x }) => x <= 100 + 1e-7));
  assert.deepEqual(nav.findPath({ x: 30, y: 70 }, { x: 300, y: 70 }), []);
});

test('large movement deltas cannot tunnel through subpixel walls or terrain gaps', () => {
  const start = { x: 10, y: 50 };
  const wall = create({ obstacles: [rectangle(80.123, 0, 80.124, 160)] });
  assert.ok(wall.move(start, { x: 200, y: 0 }).x < 80.123);
  const gap = create({ walkablePolygons: [rectangle(0, 0, 80.123, 160), rectangle(80.124, 0, 240, 160)] });
  assert.ok(gap.move(start, { x: 200, y: 0 }).x <= 80.123 + 1e-7);
});

test('movement slides along a wall, stays inside world, and does not modify input', () => {
  const nav = create({ obstacles: [rectangle(100, 0, 110, 160)] });
  const start = { x: 90, y: 20 };
  const slid = nav.move(start, { x: 50, y: 70 });
  close(slid.x, 100);
  close(slid.y, 90);
  assert.ok(nav.isWalkable(slid.x, slid.y));
  assert.deepEqual(start, { x: 90, y: 20 });
  const bound = nav.move({ x: 20, y: 140 }, { x: -100, y: 100 });
  close(bound.x, 0);
  close(bound.y, 160);
});

test('overlapping ground polygons allow passage while an invalid start cannot teleport', () => {
  const nav = create({ walkablePolygons: [rectangle(0, 0, 80, 100), rectangle(60, 60, 150, 160)] });
  assertTraversable(nav, nav.findPath({ x: 20, y: 20 }, { x: 120, y: 130 }));
  const invalid = { x: 200, y: 30 };
  assert.deepEqual(nav.findPath(invalid, { x: 20, y: 20 }), []);
  assert.deepEqual(nav.move(invalid, { x: -100, y: 0 }), invalid);
});

test('malformed map configuration fails clearly', () => {
  assert.throws(() => create({ cellSize: 0 }), RangeError);
  assert.throws(() => create({ walkablePolygons: [[[0, 0], [1, NaN], [1, 1]]] }), TypeError);
});

test('painted island paths reach every landmark from spawn and allow the return trip', () => {
  const nav = createNavigation({ ...WORLD, walkablePolygons: WALKABLE });
  assert.ok(nav.isWalkable(SPAWN.x, SPAWN.y));
  for (const [name, destination] of Object.entries(PLACEMENTS)) {
    assert.ok(nav.isWalkable(destination.x, destination.y), `${name} marker must be on the road`);
    const outward = nav.findPath(SPAWN, destination);
    assertTraversable(nav, outward);
    close(outward.at(-1).x, destination.x);
    close(outward.at(-1).y, destination.y);
    const homeward = nav.findPath(destination, SPAWN);
    assertTraversable(nav, homeward);
    close(homeward.at(-1).x, SPAWN.x);
    close(homeward.at(-1).y, SPAWN.y);
  }
});

test('island paths exclude buildings, cliff faces, campfire, water and perimeter void', () => {
  const nav = createNavigation({ ...WORLD, walkablePolygons: WALKABLE });
  const solidOrVoid = [
    [540, 210], [350, 500], [1100, 575], [1000, 666],
    [1000, 350], [1190, 397], [750, 800], [100, 100],
  ];
  for (const [x, y] of solidOrVoid) assert.equal(nav.isWalkable(x, y), false, `${x},${y} is outside the roads`);
  const down = nav.move(SPAWN, { x: 0, y: 1024 });
  assert.ok(nav.isWalkable(down.x, down.y));
  assert.ok(down.y < 750);
});
