// Check that the exported TF.js model reproduces the Keras predictions and that
// the in-browser Grad-CAM code path runs. Uses the pure-JS @tensorflow/tfjs build.
// Usage: node verify_tfjs.mjs <model_dir> <n_images> <input_size> <n_classes>
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(process.env.TFJS_FROM ?? import.meta.url);
const tf = require("@tensorflow/tfjs");

const [dir, nStr, sizeStr, kStr] = process.argv.slice(2);
const n = Number(nStr), size = Number(sizeStr), k = Number(kStr);

const modelJson = JSON.parse(readFileSync(`${dir}/model.json`, "utf8"));
const weightData = readFileSync(`${dir}/weights.bin`);
const model = await tf.loadLayersModel(tf.io.fromMemory({
  modelTopology: modelJson.modelTopology,
  weightSpecs: modelJson.weightsManifest[0].weights,
  weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength),
}));

const xBuf = readFileSync(`${dir}/x.f32`);
const x = tf.tensor4d(new Float32Array(xBuf.buffer, xBuf.byteOffset, n * size * size), [n, size, size, 1]);
const kerasBuf = readFileSync(`${dir}/keras_prob.f32`);
const keras = new Float32Array(kerasBuf.buffer, kerasBuf.byteOffset, n * k);

const t0 = performance.now();
const jsProb = await model.predict(x).data();
const batchMs = performance.now() - t0;

let maxAbs = 0, agree = 0;
for (let i = 0; i < n; i++) {
  let aj = 0, ak = 0;
  for (let c = 0; c < k; c++) {
    maxAbs = Math.max(maxAbs, Math.abs(jsProb[i * k + c] - keras[i * k + c]));
    if (jsProb[i * k + c] > jsProb[i * k + aj]) aj = c;
    if (keras[i * k + c] > keras[i * k + ak]) ak = c;
  }
  if (aj === ak) agree++;
}

// Single-image latency (as in the app) and Grad-CAM, same code as src/lib/cnn-model.ts
const times = [];
let camShape = null;
for (let i = 0; i < Math.min(n, 50); i++) {
  const one = x.slice([i, 0, 0, 0], [1, size, size, 1]);
  const s = performance.now();
  const p = model.predict(one);
  await p.data();
  times.push(performance.now() - s);
  p.dispose();
  if (i === 0) {
    const convIdx = model.layers.map((l) => l.getClassName()).lastIndexOf("Conv2D");
    const convModel = tf.model({ inputs: model.inputs, outputs: model.layers[convIdx].output });
    const head = model.layers.slice(convIdx + 1);
    const cam = tf.tidy(() => {
      const convOut = convModel.predict(one);
      const score = (c) => { let y = c; for (const l of head) y = l.apply(y); return y.gather([1], 1).sum(); };
      const g = tf.grad(score)(convOut);
      const w = g.mean([1, 2], true);
      const m = tf.relu(convOut.mul(w).sum(-1)).squeeze([0]);
      return m.div(tf.maximum(m.max(), 1e-8));
    });
    camShape = cam.shape;
    cam.dispose();
  }
  one.dispose();
}
times.sort((a, b) => a - b);
console.log(JSON.stringify({
  backend: tf.getBackend(), n, maxAbsDiff: maxAbs, top1Agreement: agree / n,
  batchMs, singleImageMedianMs: times[Math.floor(times.length / 2)], gradCamShape: camShape,
}));
