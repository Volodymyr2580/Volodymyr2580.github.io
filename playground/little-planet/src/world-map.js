// Pixel coordinates refer to the 1536 x 1024 island artwork, and to the
// explorer's feet. Keep these corridors aligned with the painted dirt paths.
export const WORLD = { width: 1536, height: 1024 };
export const SPAWN = { x: 775, y: 452 };

export const PLACEMENTS = {
  radar: { x: 606, y: 299, radius: 44, labelX: 485, labelY: 122 },
  // The ice observatory sits on a cliff with no visible stairs. Interact from
  // its dirt approach; walking on the cliff face would contradict the artwork.
  ice: { x: 899, y: 247, radius: 56, labelX: 1165, labelY: 107 },
  library: { x: 430, y: 614, radius: 44, labelX: 334, labelY: 385 },
  camp: { x: 939, y: 701, radius: 62, labelX: 1120, labelY: 479 },
};

// One polygon per road keeps navigation inexpensive. Offset the centerline
// with miter joins so neighboring road segments remain connected at bends.
function road(centerline, width) {
  const radius = width / 2;
  const normals = centerline.slice(1).map(([x, y], index) => {
    const dx = x - centerline[index][0];
    const dy = y - centerline[index][1];
    const length = Math.hypot(dx, dy);
    return [-dy / length, dx / length];
  });
  const sides = [-1, 1].map((sign) => centerline.map(([x, y], index) => {
    const before = normals[Math.max(0, index - 1)];
    const after = normals[Math.min(normals.length - 1, index)];
    let nx = before[0] + after[0];
    let ny = before[1] + after[1];
    const length = Math.hypot(nx, ny);
    nx /= length;
    ny /= length;
    const offset = radius / Math.max(0.5, nx * after[0] + ny * after[1]);
    return [x + sign * nx * offset, y + sign * ny * offset];
  }));
  return [...sides[0], ...sides[1].reverse()];
}

export const WALKABLE = [
  // Radar staircase to the central crossroads.
  road([[775, 452], [741, 415], [713, 387], [660, 369], [608, 341], [609, 316], [606, 292]], 30),
  // The path curling uphill to the ice shelf; stop before the snow cliff.
  road([[730, 389], [775, 375], [798, 348], [800, 317], [801, 302], [817, 277], [854, 258], [910, 242]], 28),
  // Main north-south path.
  road([[775, 452], [757, 498], [737, 541], [712, 575], [692, 603]], 34),
  // Library stairs and the broad curve below its gardens.
  road([[694, 600], [653, 628], [611, 642], [556, 648], [506, 640], [466, 628], [432, 615], [420, 607]], 28),
  // Camp approach ends in front of the fire, inside the perimeter fence.
  road([[693, 605], [738, 627], [784, 647], [836, 660], [880, 674], [919, 690], [950, 708]], 32),
  // Two short exploratory side paths, kept away from trees and tent ropes.
  road([[775, 452], [824, 467], [869, 490], [914, 516], [964, 538], [980, 541]], 27),
  road([[608, 340], [572, 350], [530, 357], [488, 358], [445, 350]], 23),
];
