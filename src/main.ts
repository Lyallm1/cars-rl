import { MLP, RMSProp, reduceSum } from "./autograd.js";

import { CarEnv } from "./gym.js";
import { world } from "./world.js";
import { zip } from "./commons.js";

type Transition = { state: number[], action: number, reward: number, nextState: number[], done: boolean }

const env = new CarEnv(world, 70, 180, Math.PI / 50), shape = [env.state().length, 5, 5, 3], q = new MLP(shape), qTarget = new MLP(shape), memory: Transition[] = [];
for (let nEpisodes = 0; ; nEpisodes++) {
  let game = { state: env.reset(), reward: 0, done: false }, score = 0;
  while (!game.done) {
    const act = Math.random() * 1000 > 1 + 199 * Math.exp(-nEpisodes / 200) ? q.forward(game.state).reduce(
      (i, _, j, a) => a[i].getValue() > a[j].getValue() ? i : j, 0
    ) : Math.floor(Math.random() * 3);
    const nextGame = env.step(act);
    memory.push({ state: game.state, reward: nextGame.reward, action: act, nextState: nextGame.state, done: nextGame.done });
    if(memory.length > 10000) memory.shift();
    game = nextGame;
    score += nextGame.reward;
    document.getElementById('status')!.innerText = `Episode: ${nEpisodes.toString()}, Score: ${score.toFixed(2)}`;
    env.render(document.getElementById('main-canvas') as HTMLCanvasElement);
    await new Promise(requestAnimationFrame);
  }
  if (memory.length > 1000) for (let e = 0; e < 20; e++) {
    const batch = Array.from({ length: 64 }).map(() => memory[Math.floor(Math.random() * memory.length)])
    q.zero_grad();
    reduceSum(zip(
      batch.map(s => q.forward(s.state)).map((q, i) => q[batch[i].action]),
      batch.map(s => qTarget.forward(s.nextState).reduce((a, v) => a.getValue() > v.getValue() ? a : v).mul(0.98).mul(s.done ? 0 : 1).add(s.reward))
    ).map(([q, t]) => {
      const diff = q.sub(t), absDiff = diff.abs();
      return absDiff.getValue() < 1 ? diff.mul(diff).mul(0.5) : absDiff.sub(0.5);
    })).div(64).backward();
    new RMSProp(q.parameters(), 0.01).step();
  }
  if (nEpisodes % 10 === 0) qTarget.parameters().forEach((p, i) => p.setValue(q.parameters()[i].getValue()));
}
