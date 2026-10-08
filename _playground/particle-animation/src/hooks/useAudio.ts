import { useCallback, useEffect, useRef, useState } from 'react';
import { Howl } from 'howler';

export const useAudio = (src: string, autoplay = false) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(false);
  const soundRef = useRef<Howl | null>(null);

  useEffect(() => {
    let disposed = false;

    const sound = new Howl({
      src: [src],
      loop: true,
      volume: 0.5,
      html5: true,
      preload: false,
      onloaderror: () => setError(true),
      onload: () => {
        if (autoplay && !disposed && !sound.playing()) {
          sound.play();
        }
      },
      onplay: () => { setIsPlaying(true); setError(false); },
      onpause: () => setIsPlaying(false),
      onstop: () => setIsPlaying(false),
      onplayerror: () => {
        setError(true);
        setIsPlaying(false);
        sound.once('unlock', () => {
          if (autoplay && !disposed) {
            sound.play();
          }
        });
      },
    });

    soundRef.current = sound;

    return () => {
      disposed = true;
      sound.unload();
      soundRef.current = null;
    };
  }, [src, autoplay]);

  const togglePlay = useCallback(() => {
    const sound = soundRef.current;
    if (!sound) return;

    if (sound.playing()) {
      sound.pause();
    } else {
      sound.play();
    }
  }, []);

  const setVolume = useCallback((volume: number) => {
    if (soundRef.current) {
      soundRef.current.volume(volume);
    }
  }, []);

  return { isPlaying, togglePlay, setVolume, error };
};
