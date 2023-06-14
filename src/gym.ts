import { Line, Vec, World } from "./commons.js";

import { Car } from "./car.js";

interface GymEnv<T, U> {
  reset: () => T;
  step: (action: U) => { state: T, reward: number, done: boolean };
  render: (canvas: HTMLCanvasElement) => void;
}

export const drawLine = (ctx: CanvasRenderingContext2D, line: Line, color = "#000") => {
  ctx.beginPath();
  ctx.moveTo(line.p1.x, line.p1.y);
  ctx.lineTo(line.p2.x, line.p2.y);
  ctx.strokeStyle = color;
  ctx.stroke();
}, defined = <T>(x: T | null | undefined): x is T => !x,
distance = (p1: Vec, p2: Vec) => Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2),
intersection = (l1: Line, l2: Line): Vec | null => {
  const denom = (l1.p1.x - l1.p2.x) * (l2.p1.y - l2.p2.y) - (l1.p1.y - l1.p2.y) * (l2.p1.x - l2.p2.x);
  if (denom === 0) return null;
  const x = ((l1.p1.x * l1.p2.y - l1.p1.y * l1.p2.x) * (l2.p1.x - l2.p2.x) - (l1.p1.x - l1.p2.x) * (l2.p1.x * l2.p2.y - l2.p1.y * l2.p2.x)) / denom,
  y = ((l1.p1.x * l1.p2.y - l1.p1.y * l1.p2.x) * (l2.p1.y - l2.p2.y) - (l1.p1.y - l1.p2.y) * (l2.p1.x * l2.p2.y - l2.p1.y * l2.p2.x)) / denom;
  return (x < Math.min(l1.p1.x, l1.p2.x) || x > Math.max(l1.p1.x, l1.p2.x) || x < Math.min(l2.p1.x, l2.p2.x) || x > Math.max(l2.p1.x, l2.p2.x)) ||
  (y < Math.min(l1.p1.y, l1.p2.y) || y > Math.max(l1.p1.y, l1.p2.y) || y < Math.min(l2.p1.y, l2.p2.y) || y > Math.max(l2.p1.y, l2.p2.y)) ? null : { x, y };
}

export class CarEnv implements GymEnv<number[], number> {
  private lastTime = performance.now();
  private car!: Car;
  private rewardsGrid: boolean[][] = [];

  constructor(private readonly world: World, private readonly sensorLength: number, private readonly speed: number, private readonly steerAngle: number, private readonly gridCellSize = 80) {
    this.reset();
  }

  reset() {
    this.car = new Car(this.sensorLength, this.speed, this.steerAngle);
    this.rewardsGrid = Array.from({ length: Math.ceil(this.world.width / this.gridCellSize) }, () => Array.from({ length: Math.ceil(this.world.height / this.gridCellSize) }, () => false));
    return this.state();
  }

  sensorIntersections = (): (Vec | null)[] =>  this.car.sensorsLines().map(
    sensor => this.world.walls.map(wall => intersection(sensor, wall)).filter(defined).reduce((a, v) => !a ? v : distance(this.car.center, v) < distance(this.car.center, a) ? v : a, null as Vec | null)
  )
  sensorInputs = () => this.sensorIntersections().map(itx => itx === null ? 0 : 1 - (distance(this.car.center, itx) / this.sensorLength));

  render(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, this.world.width, this.world.height);
    this.world.walls.forEach(s => drawLine(ctx, s));
    this.car.shapeLines().forEach(s => drawLine(ctx, s));
  }

  isColliding = () => this.car.shapeLines().some(line => this.world.walls.some(wall => intersection(line, wall) != null));
  state = this.sensorInputs;

  step(action: number) {
    const now = performance.now(), dt = Math.min(0.01, (now - this.lastTime) / 1000);
    if (!this.isColliding()) this.car.update(dt, action);
    this.lastTime = now;
    let pathReward = 0.01;
    const gameOver = this.isColliding(), cellX = Math.floor(this.car.center.x / this.gridCellSize), cellY = Math.floor(this.car.center.y / this.gridCellSize);
    if (!this.rewardsGrid[cellY][cellX]) {
      this.rewardsGrid[cellY][cellX] = true;
      pathReward = 1;
    }
    return { state: this.state(), reward: gameOver ? -10 : pathReward, done: gameOver };
  }
}
