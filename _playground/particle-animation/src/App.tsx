import { useState } from 'react';
import { ParticleCanvas, type GestureType } from './components/ParticleCanvas';
import { useMediaPipe } from './hooks/useMediaPipe';
import { useAudio } from './hooks/useAudio';
import type { ParticleModel } from './utils/particleData';

const models: {id: ParticleModel; zh: string; en: string}[] = [
  {id: 'heart', zh: '爱心', en: 'Heart'},
  {id: 'flower', zh: '花朵', en: 'Flower'},
  {id: 'saturn', zh: '土星', en: 'Saturn'},
  {id: 'fireworks', zh: '烟花', en: 'Fireworks'},
];

export default function App() {
  const [english, setEnglish] = useState(() => {
    try { return localStorage.getItem('site-language') === 'en'; } catch { return false; }
  });
  const t = (zh: string, en: string) => english ? en : zh;
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const camera = useMediaPipe(cameraEnabled);
  const [model, setModel] = useState<ParticleModel>('heart');
  const [manualGesture, setManualGesture] = useState<GestureType>('fist');
  const [rotation, setRotation] = useState(0);
  const [color, setColor] = useState('#f6c76b');
  const audio = useAudio(`${import.meta.env.BASE_URL}assets/bgm.mp4`, false);
  const gesture = cameraEnabled && camera.isReady && !camera.error ? camera.gesture : manualGesture;
  const handRotation = cameraEnabled && camera.isReady && !camera.error ? camera.handRotation : rotation;

  function setLanguage() {
    const language = english ? 'zh-CN' : 'en';
    setEnglish(!english);
    document.documentElement.lang = language;
    try { localStorage.setItem('site-language', language); } catch { /* Optional preference. */ }
  }

  return (
    <div className="particle-stage" lang={english ? 'en' : 'zh-CN'}>
      <ParticleCanvas gesture={gesture} handRotation={handRotation} model={model} themeColor={color} />
      <header className="stage-header">
        <div>
          <a className="back-link" href="../">← {t('互动实验室', 'Playground')}</a>
          <h1>Particle Animation</h1>
        </div>
        <button type="button" onClick={setLanguage} aria-label="Switch language / 切换语言">{english ? '中文' : 'EN'}</button>
      </header>

      <section className="stage-controls" aria-label={t('粒子控制', 'Particle controls')}>
        <div className="control-heading"><span>STARLIT PARTICLE STAGE</span><span className="live-dot" aria-hidden="true" /></div>
        <div className="model-picker">
          {models.map(option => <button type="button" key={option.id} aria-pressed={model === option.id} onClick={() => {setModel(option.id); setManualGesture('fist');}}>{english ? option.en : option.zh}</button>)}
        </div>
        <div className="gesture-picker">
          <button type="button" aria-pressed={gesture === 'fist'} onClick={() => {setCameraEnabled(false); setManualGesture('fist');}}>{t('聚合', 'Gather')}</button>
          <button type="button" aria-pressed={gesture === 'open'} onClick={() => {setCameraEnabled(false); setManualGesture('open');}}>{t('散开', 'Scatter')}</button>
          <label className="color-control">{t('颜色', 'Color')}<input aria-label={t('粒子颜色', 'Particle color')} type="color" value={color} onChange={event => setColor(event.target.value)} /></label>
        </div>
        <label className="rotation-control">{t('旋转', 'Rotation')}<input type="range" min="-1" max="1" step="0.01" value={rotation} onChange={event => {setCameraEnabled(false); setRotation(Number(event.target.value));}} /></label>
        <div className="extra-controls">
          <button type="button" aria-pressed={cameraEnabled} onClick={() => setCameraEnabled(!cameraEnabled)}>{cameraEnabled ? t('关闭摄像头', 'Stop camera') : t('开启手势控制', 'Enable gestures')}</button>
          <button type="button" aria-pressed={audio.isPlaying} onClick={audio.togglePlay}>{audio.isPlaying ? t('暂停音乐', 'Pause music') : t('播放音乐', 'Play music')}</button>
        </div>
        <p className="camera-status" role="status">{camera.error ? t('摄像头或手势模型暂不可用，请使用上方控制按钮。可关闭后重试。', 'Camera or gesture model unavailable. Use the controls above, or turn the camera off and retry.') : cameraEnabled ? camera.isReady ? t('握拳聚合 · 张手散开 · 转动手腕旋转', 'Fist to gather · Open hand to scatter · Turn wrist to rotate') : t('正在准备摄像头与手势模型…', 'Preparing camera and gesture model…') : t('无需摄像头也可体验；开启手势后，画面仅在本机处理。', 'No camera needed. Gesture video is processed on your device.')}</p>
        {audio.error && <p role="status" className="camera-status">{t('音乐暂时无法播放，请再次点击播放。', 'Audio unavailable. Click play to retry.')}</p>}
        <details><summary>{t('关于这个小实验', 'About this experiment')}</summary><p>{t('探索爱心、花朵、土星与烟花四种粒子造型。切换造型后等待片刻，让粒子慢慢聚拢。手势首次开启需要下载识别模型。', 'Explore hearts, flowers, Saturn and fireworks. Allow a moment for each transition. Gesture recognition downloads its model on first use.')}</p></details>
      </section>

      <aside className={`camera-preview ${cameraEnabled ? 'camera-visible' : ''}`} aria-label={t('摄像头预览', 'Camera preview')}>
        <video ref={camera.videoRef} playsInline muted autoPlay />
      </aside>
      <img className="stage-keepsake" src={`${import.meta.env.BASE_URL}assets/jf_2.jpg`} alt="Sherlock Holmes" />
    </div>
  );
}
