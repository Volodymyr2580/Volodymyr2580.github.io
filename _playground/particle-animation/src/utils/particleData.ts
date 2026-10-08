export type ParticleModel = 'heart' | 'flower' | 'saturn' | 'fireworks';

export interface ParticleSet {
  positions: Float32Array;
  colors: Float32Array;
}

interface Rgb {
  r: number;
  g: number;
  b: number;
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const rand = (min: number, max: number) => min + Math.random() * (max - min);

export const hexToRgb = (hex: string): Rgb => {
  const normalized = hex.replace('#', '').trim();
  const safe = normalized.length === 3
    ? normalized.split('').map((char) => char + char).join('')
    : normalized.padEnd(6, '0').slice(0, 6);

  return {
    r: parseInt(safe.slice(0, 2), 16) / 255,
    g: parseInt(safe.slice(2, 4), 16) / 255,
    b: parseInt(safe.slice(4, 6), 16) / 255,
  };
};

const mixColor = (base: Rgb, accent: Rgb, amount: number): Rgb => ({
  r: base.r * (1 - amount) + accent.r * amount,
  g: base.g * (1 - amount) + accent.g * amount,
  b: base.b * (1 - amount) + accent.b * amount,
});

const writeParticle = (
  positions: Float32Array,
  colors: Float32Array,
  index: number,
  x: number,
  y: number,
  z: number,
  color: Rgb
) => {
  const offset = index * 3;
  positions[offset] = x;
  positions[offset + 1] = y;
  positions[offset + 2] = z;
  colors[offset] = clamp01(color.r);
  colors[offset + 1] = clamp01(color.g);
  colors[offset + 2] = clamp01(color.b);
};

const makeParticleSet = (
  count: number,
  writer: (index: number, positions: Float32Array, colors: Float32Array) => void
): ParticleSet => {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  for (let index = 0; index < count; index += 1) {
    writer(index, positions, colors);
  }

  return { positions, colors };
};

export const createHeartParticles = (count: number, accent: Rgb): ParticleSet => {
  const rose = { r: 1, g: 0.28, b: 0.46 };

  return makeParticleSet(count, (index, positions, colors) => {
    const t = Math.random() * Math.PI * 2;
    const shell = Math.pow(Math.random(), 0.42);
    const xBase = 16 * Math.pow(Math.sin(t), 3);
    const yBase = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    const x = xBase * shell * 5.2 + rand(-1.4, 1.4);
    const y = (yBase - 2.2) * shell * 5.2 + rand(-1.4, 1.4);
    const z = Math.sin(t * 2) * 7 * shell + rand(-6, 6);
    const color = mixColor(rose, accent, 0.34 + Math.random() * 0.22);

    writeParticle(positions, colors, index, x, y, z, color);
  });
};

export const createFlowerParticles = (count: number, accent: Rgb): ParticleSet => {
  const blush = { r: 1, g: 0.55, b: 0.68 };
  const pollen = { r: 1, g: 0.82, b: 0.36 };

  return makeParticleSet(count, (index, positions, colors) => {
    const t = Math.random() * Math.PI * 2;
    const petal = Math.abs(Math.sin(5 * t));
    const radius = (18 + 58 * petal) * Math.pow(Math.random(), 0.36);
    const x = Math.cos(t) * radius + rand(-1, 1);
    const y = Math.sin(t) * radius + Math.sin(t * 3) * 5 + rand(-1, 1);
    const z = Math.cos(t * 5) * 8 * petal + rand(-4, 4);
    const centerBias = radius < 18 ? 0.85 : 0.18;
    const color = mixColor(mixColor(blush, pollen, centerBias), accent, 0.28);

    writeParticle(positions, colors, index, x, y, z, color);
  });
};

export const createSaturnParticles = (count: number, accent: Rgb): ParticleSet => {
  const planet = { r: 0.86, g: 0.72, b: 0.52 };
  const ring = { r: 0.96, g: 0.86, b: 0.64 };

  return makeParticleSet(count, (index, positions, colors) => {
    const isRing = Math.random() < 0.48;

    if (isRing) {
      const t = Math.random() * Math.PI * 2;
      const radius = rand(54, 112);
      const x = Math.cos(t) * radius;
      const y = Math.sin(t) * radius * 0.28;
      const z = Math.sin(t) * 20 + rand(-2.5, 2.5);
      writeParticle(positions, colors, index, x, y, z, mixColor(ring, accent, 0.3));
      return;
    }

    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(rand(-1, 1));
    const radius = 42 * Math.pow(Math.random(), 0.28);
    const x = Math.sin(phi) * Math.cos(theta) * radius;
    const y = Math.cos(phi) * radius;
    const z = Math.sin(phi) * Math.sin(theta) * radius;
    const band = 0.12 * Math.sin((y + 42) * 0.22);
    const color = mixColor(planet, accent, 0.18 + band);

    writeParticle(positions, colors, index, x, y, z, color);
  });
};

export const createFireworkParticles = (count: number, accent: Rgb): ParticleSet => {
  const spark = { r: 1, g: 0.95, b: 0.82 };

  return makeParticleSet(count, (index, positions, colors) => {
    const burstIndex = index % 5;
    const centers = [
      { x: -70, y: 28, z: 0 },
      { x: 66, y: 36, z: -4 },
      { x: -14, y: -28, z: 12 },
      { x: 24, y: 78, z: -12 },
      { x: 86, y: -42, z: 8 },
    ];
    const center = centers[burstIndex];
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(rand(-1, 1));
    const radius = rand(12, 58) * Math.pow(Math.random(), 0.18);
    const streak = Math.random() < 0.22 ? 1.35 : 1;
    const x = center.x + Math.sin(phi) * Math.cos(theta) * radius * streak;
    const y = center.y + Math.cos(phi) * radius * 0.78;
    const z = center.z + Math.sin(phi) * Math.sin(theta) * radius;
    const color = mixColor(spark, accent, 0.2 + Math.random() * 0.55);

    writeParticle(positions, colors, index, x, y, z, color);
  });
};
