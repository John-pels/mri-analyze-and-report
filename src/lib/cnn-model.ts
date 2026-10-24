"use client";

import type { CNNAnalysisResult, RegionOfInterest } from "@/types/medical-analysis";
import * as tf from "@tensorflow/tfjs";

// Runs entirely in the browser. The image never leaves the device; only the
// structured result below is sent to the server for report generation.

export type ImageType = "mri" | "ct" | "xray" | "ultrasound";

interface ModelSpec {
  url: string;
  inputSize: number;
  classes: string[];
  normalClass: string;
  version: string;
}

// Only modalities with a trained, evaluated model are listed here.
export const MODEL_SPECS: Partial<Record<ImageType, ModelSpec>> = {
  mri: {
    url: "/models/mri/model.json",
    inputSize: 64,
    classes: ["No tumour", "Glioma", "Meningioma", "Pituitary tumour"],
    normalClass: "No tumour",
    version: "scnn-mri-4class-v1",
  },
};

export function isModalitySupported(imageType: ImageType): boolean {
  return Boolean(MODEL_SPECS[imageType]);
}

const cache: Partial<Record<ImageType, Promise<tf.LayersModel>>> = {};

function loadModel(imageType: ImageType): Promise<tf.LayersModel> {
  const spec = MODEL_SPECS[imageType];
  if (!spec) {
    throw new Error(`No trained model is available for ${imageType.toUpperCase()} images yet.`);
  }
  cache[imageType] ??= tf.ready().then(() => tf.loadLayersModel(spec.url));
  return cache[imageType] as Promise<tf.LayersModel>;
}

async function decode(file: File): Promise<ImageBitmap> {
  return createImageBitmap(file);
}

// Grayscale, resize to the model input and scale to [0, 1] — matches training.
function preprocess(bitmap: ImageBitmap, size: number): tf.Tensor4D {
  return tf.tidy(() => {
    const rgb = tf.browser.fromPixels(bitmap, 3).toFloat();
    const gray = rgb.mean(2, true) as tf.Tensor3D;
    const resized = tf.image.resizeBilinear(gray, [size, size]);
    return resized.div(255).expandDims(0) as tf.Tensor4D;
  });
}

// Grad-CAM (Selvaraju et al., 2017) on the last convolutional layer.
function gradCam(model: tf.LayersModel, input: tf.Tensor4D, classIndex: number): tf.Tensor2D {
  const convIdx = model.layers.map((l) => l.getClassName()).lastIndexOf("Conv2D");
  const convModel = tf.model({ inputs: model.inputs, outputs: model.layers[convIdx].output as tf.SymbolicTensor });
  const head = model.layers.slice(convIdx + 1);

  return tf.tidy(() => {
    const convOut = convModel.predict(input) as tf.Tensor4D;
    const score = (c: tf.Tensor) => {
      let y = c;
      for (const layer of head) y = layer.apply(y) as tf.Tensor;
      return y.gather([classIndex], 1).sum();
    };
    const grads = tf.grad(score)(convOut) as tf.Tensor4D;
    const weights = grads.mean([1, 2], true);
    const cam = tf.relu(convOut.mul(weights).sum(-1)).squeeze([0]) as tf.Tensor2D;
    const max = cam.max();
    return cam.div(tf.maximum(max, 1e-8)) as tf.Tensor2D;
  });
}

// Bounding box of the Grad-CAM cells at or above `threshold` of the peak.
function boxFromCam(cam: number[][], threshold: number, width: number, height: number) {
  const rows = cam.length;
  const cols = cam[0].length;
  let minR = rows;
  let maxR = -1;
  let minC = cols;
  let maxC = -1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (cam[r][c] >= threshold) {
        minR = Math.min(minR, r);
        maxR = Math.max(maxR, r);
        minC = Math.min(minC, c);
        maxC = Math.max(maxC, c);
      }
    }
  }
  if (maxR < 0) return null;
  const sx = width / cols;
  const sy = height / rows;
  return {
    x: Math.round(minC * sx),
    y: Math.round(minR * sy),
    width: Math.round((maxC - minC + 1) * sx),
    height: Math.round((maxR - minR + 1) * sy),
  };
}

export async function analyzeWithCNN(file: File, imageType: ImageType = "mri"): Promise<CNNAnalysisResult> {
  const spec = MODEL_SPECS[imageType];
  if (!spec) {
    throw new Error(`No trained model is available for ${imageType.toUpperCase()} images yet.`);
  }
  const model = await loadModel(imageType);
  const bitmap = await decode(file);
  const input = preprocess(bitmap, spec.inputSize);

  const t0 = performance.now();
  const probsTensor = model.predict(input) as tf.Tensor2D;
  const probs = Array.from(await probsTensor.data());
  const inferenceMs = performance.now() - t0;
  probsTensor.dispose();

  const top = probs.indexOf(Math.max(...probs));
  const regionsOfInterest: RegionOfInterest[] = [];

  if (spec.classes[top] !== spec.normalClass) {
    const camTensor = gradCam(model, input, top);
    const cam = (await camTensor.array()) as number[][];
    camTensor.dispose();
    // 0.7 of the peak: on the test set the median box covers ~11% of the image.
    const box = boxFromCam(cam, 0.7, bitmap.width, bitmap.height);
    if (box) {
      regionsOfInterest.push({
        id: 1,
        location: box,
        confidence: +probs[top].toFixed(2),
        classification: `${spec.classes[top]} (Grad-CAM)`,
        intensity: "n/a",
      });
    }
  }
  input.dispose();
  bitmap.close();

  return {
    features: {
      summary: `Class probabilities from ${spec.version}`,
      dimensionality: probs.length,
      sample: probs.map((p) => +p.toFixed(4)),
    },
    predictions: {
      categories: spec.classes,
      confidenceScores: probs.map((p) => +p.toFixed(4)),
    },
    regionsOfInterest,
    analysisMetadata: {
      modelArchitecture: spec.version,
      inputDimensions: `${spec.inputSize}x${spec.inputSize}x1`,
      processingTime: `${inferenceMs.toFixed(1)}ms`,
      confidenceThreshold: 0.5,
      imageType,
    },
  };
}
