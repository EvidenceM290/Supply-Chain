// static/animated_bg.js
// Subtle animated background with moving cargos and connections

const canvas = document.getElementById('animated-bg');
const ctx = canvas.getContext('2d');

// World Link colors (red theme)
const COLORS = {
  red: '#D7263D', // World Link red
  white: '#FFFFFF',
  gray: '#DDDDDD',
  blue: '#2196F3',   // Ship
  green: '#43A047',  // Plane
  orange: '#FB8C00', // Train
  dark: '#333333'    // Truck
};

// Responsive canvas
function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// SVG path data for icons (stylized, simple)
const ICONS = {
  // Realistic ship (side view, simple cargo ship)
  ship: new Path2D('M10 40 L30 35 L90 35 L110 40 L100 45 L20 45 Z M30 35 Q35 25 45 25 Q55 25 60 35'),
  // Realistic truck (cab and trailer)
  truck: new Path2D('M10 40 H70 V30 H100 V45 H10 Z M20 45 A5 5 0 1 0 30 45 A5 5 0 1 0 20 45 M80 45 A5 5 0 1 0 90 45 A5 5 0 1 0 80 45'),
  // Realistic plane (side view, wings and tail)
  plane: new Path2D('M60 20 L110 35 L60 30 L60 50 L55 32 L10 35 L55 28 Z M60 30 L80 25'),
  // Realistic train (engine and cars)
  train: new Path2D('M10 40 H40 V30 H80 V40 H110 V45 H10 Z M20 45 A5 5 0 1 0 30 45 A5 5 0 1 0 20 45 M90 45 A5 5 0 1 0 100 45 A5 5 0 1 0 90 45'),
};

// More cargos and connections
const NUM_CARGOS = 28;
const NUM_CONNECTIONS = 22;

// Generate cargos with all four icons
const ICON_LIST = ['ship', 'truck', 'plane', 'train'];
let cargos = [];
for (let i = 0; i < NUM_CARGOS; i++) {
  const icon = ICON_LIST[i % ICON_LIST.length];
  let color;
  if (icon === 'ship') color = COLORS.blue;
  else if (icon === 'truck') color = COLORS.red;
  else if (icon === 'plane') color = COLORS.green;
  else if (icon === 'train') color = COLORS.orange;
  cargos.push({
    icon,
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    speed: 0.12 + Math.random() * 0.08,
    dir: Math.random() * 2 * Math.PI,
    color
  });
}

// Generate random connections
let connections = [];
function randomCargoIndex() { return Math.floor(Math.random() * cargos.length); }
for (let i = 0; i < NUM_CONNECTIONS; i++) {
  let a = randomCargoIndex(), b = randomCargoIndex();
  while (b === a) b = randomCargoIndex();
  connections.push([a, b]);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw connections (lines)
  ctx.save();
  ctx.globalAlpha = 0.13; // much more subtle
  ctx.strokeStyle = COLORS.red;
  ctx.lineWidth = 0.7; // even thinner
  connections.forEach(([i, j]) => {
    ctx.beginPath();
    ctx.moveTo(cargos[i].x, cargos[i].y);
    ctx.lineTo(cargos[j].x, cargos[j].y);
    ctx.stroke();
  });
  ctx.restore();

  // Draw cargos (icons)
  cargos.forEach(cargo => {
    ctx.save();
    ctx.translate(cargo.x, cargo.y);
    ctx.scale(0.38, 0.38); // even bigger icons
    ctx.strokeStyle = COLORS.gray;
    ctx.lineWidth = 2;
    ctx.fillStyle = cargo.color;
    ctx.globalAlpha = 0.55; // fainter icons
    ctx.fill(ICONS[cargo.icon]);
    ctx.stroke(ICONS[cargo.icon]);
    ctx.restore();
  });
}

function update() {
  cargos.forEach(cargo => {
    cargo.x += Math.cos(cargo.dir) * cargo.speed;
    cargo.y += Math.sin(cargo.dir) * cargo.speed;
    // Wrap around screen
    if (cargo.x < -20) cargo.x = canvas.width + 20;
    if (cargo.x > canvas.width + 20) cargo.x = -20;
    if (cargo.y < -20) cargo.y = canvas.height + 20;
    if (cargo.y > canvas.height + 20) cargo.y = -20;
  });
}

function animate() {
  update();
  draw();
  requestAnimationFrame(animate);
}
animate(); 