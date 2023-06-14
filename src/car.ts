import { Sensor, Vec } from "./commons.js";

export class Car {
  constructor(
    sensorLength = 70,private readonly speed = 100,private readonly steerAngle = Math.PI / 50,public readonly center: Vec = { x: 100, y: 120 },
    private readonly direction: Vec = { x: 1, y: 0 }, private steering = 0, private readonly sensors: Sensor[] = [
      { angle: -Math.PI / 3, size: sensorLength }, { angle: 0, size: sensorLength * 3 / 2 }, { angle: Math.PI / 3, size: sensorLength }
    ], private readonly width = 20, private readonly height = 10
  ) {}

  update(dt: number, action: number) {
    this.steering = [-this.steerAngle, 0, +this.steerAngle][action];
    this.direction.x = this.direction.x * Math.cos(this.steering) - this.direction.y * Math.sin(this.steering);
    this.direction.y = this.direction.x * Math.sin(this.steering) + this.direction.y * Math.cos(this.steering);
    this.center.x += this.direction.x * this.speed * dt / Math.sqrt(this.direction.x**2 + this.direction.y**2);
    this.center.y += this.direction.y * this.speed * dt / Math.sqrt(this.direction.x**2 + this.direction.y**2);
  }

  sensorsLines = () => this.sensors.map(sensor => ({
    p1: this.center, p2: {
      x: this.center.x + sensor.size * Math.cos(sensor.angle + Math.atan2(this.direction.y, this.direction.x)),
      y: this.center.y + sensor.size * Math.sin(sensor.angle + Math.atan2(this.direction.y, this.direction.x))
    }
  }))

  shapeLines = () => [
    { p1: { x: -this.width / 2, y: -this.height / 2 }, p2: { x: this.width / 2, y: -this.height / 2 } },
    { p1: { x: this.width / 2, y: -this.height / 2 }, p2: { x: this.width / 2, y: this.height / 2 } },
    { p1: { x: this.width / 2, y: this.height / 2 }, p2: { x: -this.width / 2, y: this.height / 2 } },
    { p1: { x: -this.width / 2, y: this.height / 2 }, p2: { x: -this.width / 2, y: -this.height / 2 } },
  ].map(line => ({
    p1: {
      x: this.center.x + line.p1.x * Math.cos(Math.atan2(this.direction.y, this.direction.x)) - line.p1.y * Math.sin(Math.atan2(this.direction.y, this.direction.x)),
      y: this.center.y + line.p1.x * Math.sin(Math.atan2(this.direction.y, this.direction.x)) + line.p1.y * Math.cos(Math.atan2(this.direction.y, this.direction.x))
    }, p2: {
      x: this.center.x + line.p2.x * Math.cos(Math.atan2(this.direction.y, this.direction.x)) - line.p2.y * Math.sin(Math.atan2(this.direction.y, this.direction.x)),
      y: this.center.y + line.p2.x * Math.sin(Math.atan2(this.direction.y, this.direction.x)) + line.p2.y * Math.cos(Math.atan2(this.direction.y, this.direction.x))
    }
  }))
}
