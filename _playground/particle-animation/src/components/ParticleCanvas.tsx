import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

import {
  createFireworkParticles,
  createFlowerParticles,
  createHeartParticles,
  createSaturnParticles,
  hexToRgb,
  type ParticleModel,
  type ParticleSet,
} from '../utils/particleData';

export type GestureType = 'fist' | 'open' | 'none';

interface ParticleCanvasProps {
  gesture: GestureType;
  handRotation: number | null;
  model: ParticleModel;
  themeColor: string;
}

type TargetKey = ParticleModel;

const PARTICLE_COUNT = window.matchMedia('(max-width: 700px)').matches ? 24000 : 64000;
const DEFAULT_ACCENT = { r: 1, g: 0.76, b: 0.42 };
const MAX_HAND_ROTATION = THREE.MathUtils.degToRad(82);
const HAND_ROTATION_DEAD_ZONE = 0.035;
const HAND_ROTATION_SMOOTHING = 0.085;
const SCATTER_SMOOTHING = 0.035;

const randomBetween = (min: number, max: number) => min + Math.random() * (max - min);

const easeInOutCubic = (value: number) => {
  const t = Math.min(1, Math.max(0, value));
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

const createRandomSet = (count: number) => {
  const randoms = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const seeds = new Float32Array(count);

  for (let i = 0; i < count; i += 1) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 2 - 1);
    const radius = 72 + Math.random() * 130;
    const offset = i * 3;

    randoms[offset] = radius * Math.sin(phi) * Math.cos(theta);
    randoms[offset + 1] = radius * Math.sin(phi) * Math.sin(theta);
    randoms[offset + 2] = radius * Math.cos(phi);
    sizes[i] = Math.random() * 1.9 + 1.1;
    seeds[i] = Math.random() * 100;
  }

  return { randoms, sizes, seeds };
};

const replaceAttribute = (
  geometry: THREE.BufferGeometry,
  name: string,
  value: Float32Array,
  itemSize: number
) => {
  geometry.setAttribute(name, new THREE.BufferAttribute(value.slice(), itemSize));
};

export const ParticleCanvas: React.FC<ParticleCanvasProps> = ({ gesture, handRotation, model, themeColor }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const starFieldRef = useRef<THREE.Points | null>(null);
  const requestRef = useRef<number>();
  const targetsRef = useRef<Record<TargetKey, ParticleSet> | null>(null);
  const activeKeyRef = useRef<TargetKey>('heart');
  const lastSetRef = useRef<ParticleSet | null>(null);
  const morphRef = useRef({ start: 0, duration: 1200, active: false });
  const burstRef = useRef({ start: -10000, duration: 1400, strength: 0 });
  const scatterRef = useRef({ current: 1.0, target: 1.0 }); // Default to scattered
  const handRotationRef = useRef({ current: 0, target: 0 });

  const [isLoading, setIsLoading] = useState(true);
  const [renderError, setRenderError] = useState(false);

  useEffect(() => {
    const mountNode = mountRef.current;
    if (!mountNode) return;

    let geometry: THREE.BufferGeometry | null = null;
    let material: THREE.ShaderMaterial | null = null;
    let renderer: THREE.WebGLRenderer | null = null;
    let disposed = false;

    const fitCamera = (camera: THREE.PerspectiveCamera) => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const compact = width <= 700;
      camera.aspect = width / height;
      camera.position.z = compact ? Math.max(350, 270 / camera.aspect) : 210;
      camera.setViewOffset(width, height, compact ? 0 : -135, compact ? height * 0.16 : 0, width, height);
      camera.updateProjectionMatrix();
    };

    const init = async () => {
      try {
      const scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x03050b, 0.006);
      sceneRef.current = scene;

      const camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.1, 1000);
      fitCamera(camera);
      cameraRef.current = camera;

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, window.innerWidth <= 700 ? 1.5 : 2));
      renderer.setClearColor(0x02030a, 1);
      mountNode.appendChild(renderer.domElement);
      rendererRef.current = renderer;

        const heart = createHeartParticles(PARTICLE_COUNT, DEFAULT_ACCENT);
        const flower = createFlowerParticles(PARTICLE_COUNT, DEFAULT_ACCENT);
        const saturn = createSaturnParticles(PARTICLE_COUNT, DEFAULT_ACCENT);
        const fireworks = createFireworkParticles(PARTICLE_COUNT, DEFAULT_ACCENT);

        // Start with a star cloud, then converge into the selected shape.
        // To represent the "star cloud", we use the random sphere points.
        const { randoms, sizes, seeds } = createRandomSet(PARTICLE_COUNT);

        // Create a 'starCloud' set that represents the initial scattered state
        const starCloudPositions = new Float32Array(randoms); // Use randoms directly as position
        const starCloudColors = new Float32Array(PARTICLE_COUNT * 3);
        for(let i=0; i<PARTICLE_COUNT; i++) {
            starCloudColors[i*3] = DEFAULT_ACCENT.r;
            starCloudColors[i*3+1] = DEFAULT_ACCENT.g;
            starCloudColors[i*3+2] = DEFAULT_ACCENT.b;
        }

        const starCloud: ParticleSet = {
            positions: starCloudPositions,
            colors: starCloudColors
        };

        if (disposed) return;

        targetsRef.current = {
          heart,
          flower,
          saturn,
          fireworks,
        };

        // Start from starCloud state
        lastSetRef.current = starCloud;

        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(starCloud.positions.slice(), 3));
        geometry.setAttribute('posFrom', new THREE.BufferAttribute(starCloud.positions.slice(), 3));
        geometry.setAttribute('posTo', new THREE.BufferAttribute(starCloud.positions.slice(), 3));
        geometry.setAttribute('colorFrom', new THREE.BufferAttribute(starCloud.colors.slice(), 3));
        geometry.setAttribute('colorTo', new THREE.BufferAttribute(starCloud.colors.slice(), 3));
        geometry.setAttribute('randoms', new THREE.BufferAttribute(randoms, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('seed', new THREE.BufferAttribute(seeds, 1));

        // Start with a scattered state (u_scatter = 1.0)
        scatterRef.current = { current: 1.0, target: 1.0 };

        material = new THREE.ShaderMaterial({
          uniforms: {
            time: { value: 0 },
            u_morph: { value: 1 },
            u_scatter: { value: 1.0 }, // Initialize as fully scattered
            u_themeColor: { value: new THREE.Color(DEFAULT_ACCENT.r, DEFAULT_ACCENT.g, DEFAULT_ACCENT.b) },
            u_accentStrength: { value: 0.48 },
          },
          vertexShader: `
            uniform float time;
            uniform float u_morph;
            uniform float u_scatter;

            attribute vec3 posFrom;
            attribute vec3 posTo;
            attribute vec3 colorFrom;
            attribute vec3 colorTo;
            attribute vec3 randoms;
            attribute float size;
            attribute float seed;

            varying vec3 vColor;
            varying float vGlow;
            varying float vScatter;

            void main() {
              vec3 formed = mix(posFrom, posTo, u_morph);
              float drift = sin(time * 0.75 + seed) * 0.9 + cos(time * 0.42 + seed * 0.7) * 0.7;

              // Only apply breath and swirl to the scattered state (u_scatter > 0)
              // This makes the formed text perfectly stable and sharp
              vec3 breath = vec3(
                sin(time * 0.5 + seed) * 0.9,
                cos(time * 0.48 + seed * 1.3) * 0.8,
                drift
              ) * u_scatter;

              vec3 swirl = normalize(vec3(-formed.y, formed.x, randoms.z * 0.22) + 0.0001) * u_scatter * 16.0;
              vec3 scattered = formed + randoms * u_scatter + swirl + breath;

              vec4 mvPosition = modelViewMatrix * vec4(scattered, 1.0);
              float depth = max(24.0, length(mvPosition.xyz));

              // Sparkle effect also only applies to scattered particles, text should be solid
              float sparkle = mix(1.0, 0.72 + 0.28 * sin(time * 2.2 + seed * 6.2831), u_scatter);

              vColor = mix(colorFrom, colorTo, u_morph);
              vGlow = sparkle + u_scatter * 0.35;
              vScatter = u_scatter;

              // Base size for text is smaller to increase sharpness
              float baseSize = mix(size * 0.28, size, u_scatter);

              gl_PointSize = baseSize * sparkle * (360.0 / depth) * (1.0 + u_scatter * 0.55);
              gl_PointSize = clamp(gl_PointSize, 1.0, 6.0);
              gl_Position = projectionMatrix * mvPosition;
            }
          `,
          fragmentShader: `
            uniform vec3 u_themeColor;
            uniform float u_accentStrength;

            varying vec3 vColor;
            varying float vGlow;
            varying float vScatter;

            void main() {
              vec2 uv = gl_PointCoord.xy - vec2(0.5);
              float dist = length(uv);
              if (dist > 0.5) discard;

              float core = smoothstep(0.46, 0.08, dist);
              float halo = smoothstep(0.5, 0.0, dist) * mix(0.08, 0.45, vScatter);
              vec3 warmed = mix(vColor, u_themeColor, u_accentStrength * (0.22 + halo));
              vec3 finalColor = warmed * mix(1.06, 1.1 + vGlow * 0.58, vScatter);
              float alpha = (core * mix(0.98, 0.86, vScatter) + halo * mix(0.12, 0.34, vScatter)) * 0.94;

              gl_FragColor = vec4(finalColor, alpha);
            }
          `,
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });

        const particles = new THREE.Points(geometry, material);
        particles.frustumCulled = false;
        scene.add(particles);
        particlesRef.current = particles;

        const stars = createStarField();
        scene.add(stars);
        starFieldRef.current = stars;

        setIsLoading(false);

      } catch (error) {
        setRenderError(true);
        setIsLoading(false);
      }
    };

    const createStarField = () => {
      const count = 1700;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      const sizes = new Float32Array(count);
      const color = new THREE.Color('#f8dfaa');

      for (let i = 0; i < count; i += 1) {
        const offset = i * 3;
        positions[offset] = (Math.random() - 0.5) * 420;
        positions[offset + 1] = (Math.random() - 0.5) * 250;
        positions[offset + 2] = -90 - Math.random() * 190;
        colors[offset] = color.r * randomBetween(0.58, 1);
        colors[offset + 1] = color.g * randomBetween(0.58, 1);
        colors[offset + 2] = color.b * randomBetween(0.58, 1);
        sizes[i] = randomBetween(0.4, 1.3);
      }

      const starGeometry = new THREE.BufferGeometry();
      starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      starGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      starGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

      const starMaterial = new THREE.PointsMaterial({
        size: 0.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.54,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });

      return new THREE.Points(starGeometry, starMaterial);
    };

    init();

    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return;
      fitCamera(cameraRef.current);
      rendererRef.current.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    const startTime = performance.now();
    const animate = () => {
      if (document.hidden) {
        requestRef.current = requestAnimationFrame(animate);
        return;
      }
      const now = performance.now();
      const elapsedTime = (now - startTime) / 1000;

      if (particlesRef.current) {
        const shader = particlesRef.current.material as THREE.ShaderMaterial;
        const morph = morphRef.current;

        if (morph.active) {
          const rawProgress = (now - morph.start) / morph.duration;
          shader.uniforms.u_morph.value = easeInOutCubic(rawProgress);

          if (rawProgress >= 1) {
            shader.uniforms.u_morph.value = 1;
            morph.active = false;
          }
        }

        const burst = burstRef.current;
        const burstAge = (now - burst.start) / burst.duration;
        const burstCurve = burstAge >= 0 && burstAge <= 1
          ? Math.sin(Math.PI * burstAge)
          : 0;
        const scatter = scatterRef.current;
        scatter.current += (scatter.target - scatter.current) * SCATTER_SMOOTHING;

        shader.uniforms.time.value = elapsedTime;
        shader.uniforms.u_scatter.value = Math.min(
          1.48,
          scatter.current + Math.pow(burstCurve, 1.45) * burst.strength
        );

        const scene = sceneRef.current;
        if (scene) {
          const handRotation = handRotationRef.current;
          handRotation.current += (handRotation.target - handRotation.current) * HAND_ROTATION_SMOOTHING;

          scene.rotation.y = Math.sin(elapsedTime * 0.16) * 0.09 + handRotation.current;
          scene.rotation.x = Math.cos(elapsedTime * 0.12) * 0.045;
        }
      }

      if (starFieldRef.current) {
        starFieldRef.current.rotation.z = elapsedTime * 0.006;
        starFieldRef.current.rotation.y = Math.sin(elapsedTime * 0.08) * 0.05;
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }

      requestRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      disposed = true;
      window.removeEventListener('resize', handleResize);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (renderer && mountNode.contains(renderer.domElement)) mountNode.removeChild(renderer.domElement);
      geometry?.dispose();
      material?.dispose();
      starFieldRef.current?.geometry.dispose();
      (starFieldRef.current?.material as THREE.Material | undefined)?.dispose();
      renderer?.dispose();
    };
  }, []);

  useEffect(() => {
    if (handRotation === null || Math.abs(handRotation) < HAND_ROTATION_DEAD_ZONE) {
      handRotationRef.current.target = 0;
      return;
    }

    const rotationInput =
      (Math.abs(handRotation) - HAND_ROTATION_DEAD_ZONE) / (1 - HAND_ROTATION_DEAD_ZONE);
    handRotationRef.current.target = Math.sign(handRotation) * rotationInput * MAX_HAND_ROTATION;
  }, [handRotation, model]);

  useEffect(() => {
    const rgb = hexToRgb(themeColor);
    const material = particlesRef.current?.material as THREE.ShaderMaterial | undefined;
    if (!material) return;
    material.uniforms.u_themeColor.value.setRGB(rgb.r, rgb.g, rgb.b);
  }, [themeColor]);

  useEffect(() => {
    const targets = targetsRef.current;
    const particles = particlesRef.current;
    if (!targets || !particles) return;

    const key: TargetKey = model;

    if (gesture === 'none') return;

    // Open -> scatter back to Star Cloud.
    // Fist -> converge to the selected shape.

    // We create a temporary set to represent the scattered Star Cloud
    const starCloudSet: ParticleSet = {
        positions: particles.geometry.attributes.randoms.array as Float32Array,
        colors: particles.geometry.attributes.colorFrom.array as Float32Array // Keep current colors
    };

    const isScattering = gesture === 'open';
    const nextSet = isScattering ? starCloudSet : targets[key];
    const fromSet = lastSetRef.current ?? starCloudSet;

    if (nextSet === lastSetRef.current) return;

    // Set scatter target based on gesture
    scatterRef.current.target = isScattering ? 1.0 : 0.0;

    const bufferGeometry = particles.geometry;

    replaceAttribute(bufferGeometry, 'posFrom', fromSet.positions, 3);
    replaceAttribute(bufferGeometry, 'posTo', nextSet.positions, 3);
    replaceAttribute(bufferGeometry, 'colorFrom', fromSet.colors, 3);
    replaceAttribute(bufferGeometry, 'colorTo', nextSet.colors, 3);
    replaceAttribute(bufferGeometry, 'position', nextSet.positions, 3);

    morphRef.current = {
      start: performance.now(),
      duration: isScattering ? 3000 : 2200,
      active: true,
    };

    // We can add a slight burst effect when converging (fist)
    burstRef.current = {
      start: performance.now(),
      duration: 1500,
      strength: gesture === 'fist' ? 0.45 : 0.0,
    };

    activeKeyRef.current = key;
    lastSetRef.current = nextSet;
  }, [gesture, model]);

  return (
    <>
      {renderError && <div className="render-error" role="alert">当前浏览器无法启动粒子画面，请使用支持 WebGL 的浏览器。<br />Unable to start WebGL. Please try a compatible browser.<br /><a href="../">← Playground</a></div>}
      {isLoading && (
        <div className="pointer-events-none absolute inset-x-0 top-1/2 z-20 flex -translate-y-1/2 items-center justify-center">
          <div className="rounded-full border border-amber-200/20 bg-black/30 px-5 py-3 text-sm font-medium tracking-[0.24em] text-amber-100/80 shadow-[0_0_48px_rgba(245,198,104,0.18)] backdrop-blur-xl">
            WEAVING PARTICLES
          </div>
        </div>
      )}
      <div ref={mountRef} className="pointer-events-none absolute inset-0 z-0 bg-[#02030a]" />
    </>
  );
};
