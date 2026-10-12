// Screenshot Perfetto UI views of cropped magpie traces through headless Chromium (CDP).
//   node tools/shots.mjs out/trace/crops/plan.json
// Needs chromium with --remote-debugging-port=9333 and the crops served on http://localhost:8080 (see trace_shots.py).
import {readFileSync, writeFileSync} from 'node:fs';

const CDP = 'http://127.0.0.1:9333';
const TRACE_BASE = 'http://localhost:8080';
const WIDTH = 1920;
const HEIGHT = 1080;
const plan = JSON.parse(readFileSync(process.argv[2], 'utf8'));

// Minimal CDP client: sends commands over one websocket and resolves each reply by id.
class Session {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const {resolve, reject} = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    };
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.ws.send(JSON.stringify({id, method, params}));
    });
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Attach to the first page target chromium opened.
async function connect() {
  const list = await (await fetch(`${CDP}/json/list`)).json();
  const target = list.find((t) => t.type === 'page');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = rej;
  });
  return new Session(ws);
}

// Perfetto deep link: load the crop and expand every track group at startup.
function perfettoUrl(shot) {
  const params = new URLSearchParams({
    url: `${TRACE_BASE}/${shot.file}`,
    startupCommands: JSON.stringify([{id: 'dev.perfetto.ExpandTracksByRegex', args: ['.*']}]),
  });
  return `https://ui.perfetto.dev/#!/?${params}`;
}

// Reports whether the trace is parsed (CURRENT TRACE shown) and how many canvases have painted.
const READY_PROBE = `(() => {
  const text = document.body.innerText;
  return JSON.stringify({
    loaded: /CURRENT TRACE/.test(text),
    loading: /Loading trace/.test(text),
    canvases: document.querySelectorAll('canvas').length,
  });
})()`;

// Closes first-run dialogs.
const DISMISS = `[...document.querySelectorAll('button,div[role=button],a')]
  .filter((e) => /^(OK|Got it|Accept|I agree)$/.test(e.textContent.trim()))
  .forEach((e) => e.click())`;

// Hides the sidebar and expands the track tree so the GPU annotation and kernel rows are on screen.
const ROOMY = `for (const t of ['Hide sidebar', 'Expand all']) {
  document.querySelector('button[title="' + t + '"]')?.click();
}`;

// Polls until the trace is parsed and painted; throws after about 90 s.
async function waitReady(session, name) {
  for (let attempt = 0; attempt < 60; attempt++) {
    await sleep(1500);
    const res = await session.send('Runtime.evaluate', {expression: READY_PROBE, returnByValue: true});
    const state = JSON.parse(res.result.value);
    if (state.loaded && !state.loading && state.canvases >= 3) return state;
  }
  throw new Error(`timeline never loaded for ${name}`);
}

// Loads one crop in a fresh page, tidies the UI, waits for idle, and writes its PNG.
async function shoot(session, shot) {
  await session.send('Page.navigate', {url: 'about:blank'});
  await sleep(400);
  await session.send('Page.navigate', {url: perfettoUrl(shot)});
  const state = await waitReady(session, shot.name);
  await session.send('Runtime.evaluate', {expression: DISMISS});
  await session.send('Runtime.evaluate', {expression: ROOMY});
  await session.send('Runtime.evaluate', {
    expression: 'window.waitForPerfettoIdle ? window.waitForPerfettoIdle() : 0',
    awaitPromise: true,
    returnByValue: true,
  });
  await sleep(2000);
  const png = await session.send('Page.captureScreenshot', {format: 'png'});
  writeFileSync(`${plan.out}/${shot.name}.png`, Buffer.from(png.data, 'base64'));
  console.log(`shot ${shot.name} ${JSON.stringify(state)}`);
}

const session = await connect();
await session.send('Page.enable');
await session.send('Runtime.enable');
await session.send('Emulation.setDeviceMetricsOverride',
    {width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false});
for (const shot of plan.shots) {
  await shoot(session, shot);
}
process.exit(0);
