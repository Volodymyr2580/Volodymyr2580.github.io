import { useCallback, useEffect, useRef, useState } from 'react';
import type { Hands, Results } from '@mediapipe/hands';


type GestureType = 'fist' | 'open' | 'none';
type HandLandmark = { x: number; y: number; z?: number };
type HandLandmarks = HandLandmark[];

const HAND_ROTATION_SMOOTHING = 0.34;
const HAND_ROTATION_UPDATE_THRESHOLD = 0.018;

const CONFIRM_FRAMES: Record<GestureType, number> = {
  fist: 5,
  open: 7,
  none: 10,
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const amplifyRotation = (value: number) => {
  const clamped = clamp(value, -1, 1);
  return Math.sign(clamped) * Math.pow(Math.abs(clamped), 0.58);
};

const distance = (a: HandLandmark, b: HandLandmark) => {
  const zDelta = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(a.x - b.x, a.y - b.y, zDelta);
};

const getPalmCenterX = (landmarks: HandLandmarks) => {
  const palmIndexes = [0, 5, 9, 13, 17];
  const total = palmIndexes.reduce((sum, index) => sum + landmarks[index].x, 0);
  return total / palmIndexes.length;
};

const subtract = (a: HandLandmark, b: HandLandmark) => ({
  x: a.x - b.x,
  y: a.y - b.y,
  z: (a.z ?? 0) - (b.z ?? 0),
});

const cross = (a: HandLandmark, b: HandLandmark) => ({
  x: a.y * (b.z ?? 0) - (a.z ?? 0) * b.y,
  y: (a.z ?? 0) * b.x - a.x * (b.z ?? 0),
  z: a.x * b.y - a.y * b.x,
});

const getPalmNormalTurn = (landmarks: HandLandmarks) => {
  const wrist = landmarks[0];
  const indexVector = subtract(landmarks[5], wrist);
  const pinkyVector = subtract(landmarks[17], wrist);
  const normal = cross(indexVector, pinkyVector);
  const horizontalNormal = Math.hypot(normal.x, normal.z ?? 0);

  if (horizontalNormal < 0.000001) return 0;

  return clamp(normal.x / horizontalNormal, -1, 1);
};

const getWristTurn = (landmarks: HandLandmarks, worldLandmarks?: HandLandmarks) => {
  const source = worldLandmarks ?? landmarks;
  const indexKnuckle = source[5];
  const pinkyKnuckle = source[17];
  const palmWidth = Math.max(0.001, distance(indexKnuckle, pinkyKnuckle));
  const depthTurn = clamp(((pinkyKnuckle.z ?? 0) - (indexKnuckle.z ?? 0)) / palmWidth, -1, 1);
  const normalTurn = getPalmNormalTurn(source);
  const rollTurn = clamp((pinkyKnuckle.y - indexKnuckle.y) / palmWidth, -1, 1);

  return clamp(depthTurn * 1.15 + normalTurn * 0.9 + rollTurn * 0.22, -1, 1);
};

const getRotationControl = (landmarks: HandLandmarks, worldLandmarks?: HandLandmarks) => {
  const positionTurn = clamp((0.5 - getPalmCenterX(landmarks)) * 2, -1, 1);
  const wristTurn = getWristTurn(landmarks, worldLandmarks);

  return amplifyRotation(clamp(positionTurn * 0.2 + wristTurn * 1.45, -1, 1));
};

const isFingerExtended = (landmarks: HandLandmarks, tip: number, pip: number, mcp: number) => {
  const wrist = landmarks[0];
  const tipFromWrist = distance(landmarks[tip], wrist);
  const pipFromWrist = distance(landmarks[pip], wrist);
  const tipFromBase = distance(landmarks[tip], landmarks[mcp]);
  const pipFromBase = distance(landmarks[pip], landmarks[mcp]);

  return tipFromWrist > pipFromWrist * 1.08 && tipFromBase > pipFromBase * 1.15;
};

const classifyGesture = (landmarks: HandLandmarks): GestureType => {
  const extendedCount = [
    isFingerExtended(landmarks, 8, 6, 5),
    isFingerExtended(landmarks, 12, 10, 9),
    isFingerExtended(landmarks, 16, 14, 13),
    isFingerExtended(landmarks, 20, 18, 17),
  ].filter(Boolean).length;

  if (extendedCount >= 3) return 'open';
  if (extendedCount <= 1) return 'fist';
  return 'none';
};

export const useMediaPipe = (enabled = false) => {
  const [gesture, setGesture] = useState<GestureType>('none');
  const [handRotation, setHandRotation] = useState<number | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const latestGestureRef = useRef<GestureType>('none');
  const pendingGestureRef = useRef<GestureType>('none');
  const pendingGestureFramesRef = useRef(0);
  const smoothedHandRotationRef = useRef<number | null>(null);
  const latestHandRotationRef = useRef<number | null>(null);

  const updateGesture = useCallback((candidate: GestureType) => {
    if (candidate === latestGestureRef.current) {
      pendingGestureRef.current = candidate;
      pendingGestureFramesRef.current = 0;
      return;
    }

    if (candidate !== pendingGestureRef.current) {
      pendingGestureRef.current = candidate;
      pendingGestureFramesRef.current = 1;
      return;
    }

    pendingGestureFramesRef.current += 1;

    if (pendingGestureFramesRef.current < CONFIRM_FRAMES[candidate]) return;

    latestGestureRef.current = candidate;
    pendingGestureFramesRef.current = 0;
    setGesture(candidate);
  }, []);

  const updateHandRotation = useCallback((nextValue: number | null) => {
    if (nextValue === null) {
      smoothedHandRotationRef.current = null;
      latestHandRotationRef.current = null;
      setHandRotation(null);
      return;
    }

    const previousSmoothed = smoothedHandRotationRef.current;
    const smoothed =
      previousSmoothed === null
        ? nextValue
        : previousSmoothed + (nextValue - previousSmoothed) * HAND_ROTATION_SMOOTHING;

    smoothedHandRotationRef.current = smoothed;

    const previousPublished = latestHandRotationRef.current;
    const shouldPublish =
      previousPublished === null || Math.abs(previousPublished - smoothed) > HAND_ROTATION_UPDATE_THRESHOLD;

    if (!shouldPublish) return;

    latestHandRotationRef.current = smoothed;
    setHandRotation(smoothed);
  }, []);

  useEffect(() => {
    setIsReady(false);
    setError(false);
    setGesture('none');
    setHandRotation(null);
    latestGestureRef.current = 'none';
    pendingGestureRef.current = 'none';
    pendingGestureFramesRef.current = 0;
    smoothedHandRotationRef.current = null;
    latestHandRotationRef.current = null;
    if (!enabled || !videoRef.current) return;

    const video = videoRef.current;
    let disposed = false;
    let stream: MediaStream | undefined;
    let hands: Hands | undefined;
    let frame = 0;
    const stopTracks = () => stream?.getTracks().forEach(track => track.stop());
    const dispose = () => {
      disposed = true;
      cancelAnimationFrame(frame);
      stopTracks();
      video.srcObject = null;
      void hands?.close().catch(() => {});
    };
    const fail = () => {
      if (!disposed) {
        setError(true);
        setIsReady(false);
        dispose();
      }
    };
    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480, facingMode: 'user'}, audio: false});
        if (disposed) { stopTracks(); return; }
        video.srcObject = stream;
        await video.play();
        const module = await import('@mediapipe/hands');
        if (disposed) return;
        hands = new module.Hands({
          locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
        });
        hands.setOptions({maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.64, minTrackingConfidence: 0.64});
        hands.onResults((results: Results) => {
          if (disposed) return;
          const landmarks = results.multiHandLandmarks?.[0] as HandLandmarks | undefined;
          if (landmarks) {
            updateHandRotation(getRotationControl(landmarks, results.multiHandWorldLandmarks?.[0] as HandLandmarks | undefined));
            updateGesture(classifyGesture(landmarks));
          } else {
            updateHandRotation(null);
            updateGesture('none');
          }
        });
        let timeout: ReturnType<typeof setTimeout>;
        try {
          await Promise.race([hands.initialize(), new Promise<never>((_, reject) => {
            timeout = setTimeout(() => reject(new Error('Model load timed out')), 30000);
          })]);
        } finally { clearTimeout(timeout!); }
        if (disposed) { void hands.close().catch(() => {}); return; }
        setIsReady(true);
        const tick = async () => {
          if (disposed) return;
          try {
            if (!document.hidden && video.readyState >= 2) await hands!.send({image: video});
            if (!disposed) frame = requestAnimationFrame(tick);
          } catch { fail(); }
        };
        void tick();
      } catch { fail(); }
    };
    window.addEventListener('pagehide', dispose);
    void start();
    return () => { window.removeEventListener('pagehide', dispose); dispose(); };
  }, [enabled, updateGesture, updateHandRotation]);

  return { gesture, handRotation, isReady, error, videoRef };
};
