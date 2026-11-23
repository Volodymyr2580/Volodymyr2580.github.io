// Public site destinations verified against the corresponding Jekyll front matter.
// Keep scene positions and interaction logic separate from this bilingual copy.
export const PLACES = [
  {
    id: 'radar',
    content: {
      zh: {
        title: '回声雷达站',
        kicker: '01 · 地下的线索',
        summary: '发出一个信号，听听地层的回答。',
        body: '我关注探地雷达全波形反演，以及强化学习在地震反演中的应用：如何从观测到的波形出发，理解看不见的地下结构？\n\n这里的小雷达把“发射—传播—反射”画成一段动画。它是定性的概念示意，不是数值正演；回波的间隔、亮度和层位不代表真实测量或反演结果。',
        links: [
          { label: '走进全波形反演', href: '/research/fwi/' },
          { label: '探地雷达研究', href: '/research/ground-penetrating-radar/' },
          { label: '强化学习与反演', href: '/research/reinforcement-learning/' },
        ],
      },
      en: {
        title: 'Echo Radar Station',
        kicker: '01 · Beneath the surface',
        summary: 'Send a signal. Listen for the layers below.',
        body: 'I explore full waveform inversion for ground-penetrating radar and reinforcement learning for seismic inversion. How can recorded waveforms help us understand structures we cannot see?\n\nThe little radar illustrates transmission, propagation, and reflection. This is a qualitative animation, not a numerical forward simulation. Echo timing, brightness, and layer positions do not represent measurements or inversion results.',
        links: [
          { label: 'Full waveform inversion', href: '/research/fwi/' },
          { label: 'Ground-penetrating radar', href: '/research/ground-penetrating-radar/' },
          { label: 'Reinforcement learning & inversion', href: '/research/reinforcement-learning/' },
        ],
      },
    },
  },
  {
    id: 'ice',
    content: {
      zh: {
        title: '极地观测营',
        kicker: '02 · 遥远世界的回声',
        summary: '把好奇心带到月球、火星与极地冰层。',
        body: '我的行星科学兴趣目前主要围绕月球与火星。雷达提供了一种观察地表以下结构的方式，也带来许多关于冰层、介质与波传播的问题。\n\n这片冰原是小星球里的想象场景。你可以从这里回到研究兴趣介绍，或继续了解探地雷达数据的处理与解释。',
        links: [
          { label: '我的研究兴趣', href: '/#home-research-title' },
          { label: '了解雷达研究', href: '/research/ground-penetrating-radar/' },
        ],
      },
      en: {
        title: 'Polar Observatory',
        kicker: '02 · Echoes from distant worlds',
        summary: 'A little curiosity for the Moon, Mars, and polar ice.',
        body: 'My planetary science interests currently focus on the Moon and Mars. Radar offers a way to investigate the subsurface, raising questions about ice layers, material properties, and wave propagation.\n\nThis icy corner is an imagined landscape. Follow the paths below to my research interests or an introduction to processing and interpreting ground-penetrating radar observations.',
        links: [
          { label: 'My research interests', href: '/#home-research-title' },
          { label: 'Explore radar research', href: '/research/ground-penetrating-radar/' },
        ],
      },
    },
  },
  {
    id: 'library',
    content: {
      zh: {
        title: '林间书屋',
        kicker: '03 · 随手翻开一页',
        summary: '数学、机器学习，还有一路写下的想法。',
        body: '有些问题从一行公式开始，有些从一段阅读开始。我把数学、机器学习和地球物理的学习记录留在笔记里，也在博客中写下思考。\n\n找个安静的位置，挑一页开始读吧。你可以随时回来，继续探索这颗小星球。',
        links: [
          { label: '打开学习笔记', href: '/notes/' },
          { label: '阅读博客', href: '/blogs/' },
        ],
      },
      en: {
        title: 'Woodland Library',
        kicker: '03 · Open a page',
        summary: 'Mathematics, machine learning, and thoughts along the way.',
        body: 'Some questions begin with an equation; others begin with a book. My notes collect studies in mathematics, machine learning, and geophysics, while my blog leaves room for reflections.\n\nFind a quiet spot and choose a page. The planet will be here when you return.',
        links: [
          { label: 'Browse study notes', href: '/notes/' },
          { label: 'Read the blog', href: '/blogs/' },
        ],
      },
    },
  },
  {
    id: 'camp',
    content: {
      zh: {
        title: '星光营地',
        kicker: '04 · 很高兴在这里遇见你',
        summary: "欢迎来到 Volodymyr's Planet。",
        body: '你好，我是 Volodymyr。目前是北京大学地球与空间科学学院的博士生，关注地球物理、深度学习、行星科学与科普。\n\n研究之外，我喜欢小说、电影、游戏、运动和做饭，也喜欢猫与手工。来营火旁坐坐；想聊研究或打个招呼，都可以给我写信。',
        links: [
          { label: '认识我', href: '/' },
          { label: '学术产出', href: '/publications/' },
          { label: '写一封信', href: 'mailto:2200012437@stu.pku.edu.cn' },
        ],
      },
      en: {
        title: 'Starlight Camp',
        kicker: '04 · Good to see you here',
        summary: "Welcome to Volodymyr's Planet.",
        body: "Hi, I'm Volodymyr, or Wenzhe Sheng. I'm a PhD student at Peking University's School of Earth and Space Sciences, interested in geophysics, deep learning, planetary science, and science communication.\n\nAway from research, I enjoy fiction, films, games, sports, cooking, cats, and making things by hand. Take a seat by the fire. If you would like to talk research or say hello, send me a letter.",
        links: [
          { label: 'About me', href: '/' },
          { label: 'Academic work', href: '/publications/' },
          { label: 'Send a letter', href: 'mailto:2200012437@stu.pku.edu.cn' },
        ],
      },
    },
  },
];

export const TEXT = {
  zh: {
    title: "Volodymyr's Planet",
    subtitle: '一颗关于好奇心的小星球',
    back: '返回 Playground',
    explore: '开始探索',
    move: 'WASD / 方向键移动',
    touchMove: '拖动摇杆移动',
    interact: 'E / 空格互动',
    interactButton: '看看这里',
    close: '继续探索',
    places: '地点图鉴',
    visited: '已探索',
    unvisited: '待探索',
    allPlaces: '查看所有地点',
    radarStart: '发射探测波',
    radarCaption: '定性示意 · 非数值正演',
    radarRunning: '正在聆听回声…',
    resetPosition: '回到营地',
    soundOn: '打开声音',
    soundOff: '关闭声音',
    loading: '小星球正在苏醒…',
    error: '场景暂时无法加载，仍可通过地点图鉴浏览内容。',
    completed: '所有地点都留下了你的足迹。',
  },
  en: {
    title: "Volodymyr's Planet",
    subtitle: 'A tiny world of curiosity',
    back: 'Back to Playground',
    explore: 'Start exploring',
    move: 'WASD / arrows to move',
    touchMove: 'Drag the joystick to move',
    interact: 'E / Space to interact',
    interactButton: 'Take a look',
    close: 'Keep exploring',
    places: 'Field guide',
    visited: 'Explored',
    unvisited: 'To discover',
    allPlaces: 'Browse all places',
    radarStart: 'Send a radar pulse',
    radarCaption: 'Qualitative illustration · not a numerical simulation',
    radarRunning: 'Listening for echoes…',
    resetPosition: 'Return to camp',
    soundOn: 'Turn sound on',
    soundOff: 'Turn sound off',
    loading: 'Waking up the planet…',
    error: 'The scene could not load. You can still browse the field guide.',
    completed: 'Your footprints now reach every place.',
  },
};
