import * as tf from "@tensorflow/tfjs";

// Model state
const models: Record<string, tf.LayersModel | null> = {
  mri: null,
  ct: null,
  xray: null,
  ultrasound: null,
};

const modelLoaded: Record<string, boolean> = {
  mri: false,
  ct: false,
  xray: false,
  ultrasound: false,
};

const modelLoading: Record<string, boolean> = {
  mri: false,
  ct: false,
  xray: false,
  ultrasound: false,
};

// Initialize and load the CNN model for a specific image type
async function loadModel(imageType: string) {
  if (!Object.keys(models).includes(imageType)) {
    throw new Error(`Unsupported image type: ${imageType}`);
  }

  if (modelLoaded[imageType] || modelLoading[imageType]) return;

  modelLoading[imageType] = true;
  try {
    // Create a new CNN model for the specific image type
    models[imageType] = await createMedicalImageModel(imageType);

    // In a production app, you would load a pre-trained model instead:
    // models[imageType] = await tf.loadLayersModel(`path/to/${imageType}_model.json`);

    console.log(`CNN model for ${imageType} loaded successfully`);
    modelLoaded[imageType] = true;
  } catch (error) {
    console.error(`Failed to load CNN model for ${imageType}:`, error);
    throw new Error(`Failed to initialize the CNN model for ${imageType}`);
  } finally {
    modelLoading[imageType] = false;
  }
}

// Create a CNN model architecture for different medical image types
async function createMedicalImageModel(
  imageType: string
): Promise<tf.LayersModel> {
  // This is a simplified CNN architecture for demonstration
  // In a real application, you would use a more sophisticated architecture
  // or a pre-trained model fine-tuned on specific medical image data

  const model = tf.sequential();

  // Common base architecture with some type-specific adjustments
  const inputChannels = imageType === "mri" || imageType === "ct" ? 1 : 3;
  const initialFilters = getInitialFilters(imageType);

  // Input layer
  model.add(
    tf.layers.conv2d({
      inputShape: [224, 224, inputChannels],
      filters: initialFilters,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      name: "conv1",
    })
  );

  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool1" }));

  // Add more convolutional layers based on image type
  const layerConfig = getLayerConfig(imageType);

  for (let i = 0; i < layerConfig.length; i++) {
    const config = layerConfig[i];
    const layerIndex = i + 2; // Start from 2 since we already have conv1 and pool1

    model.add(
      tf.layers.conv2d({
        filters: config.filters,
        kernelSize: config.kernelSize,
        activation: "relu",
        padding: "same",
        name: `conv${layerIndex}`,
      })
    );

    model.add(
      tf.layers.maxPooling2d({ poolSize: 2, name: `pool${layerIndex}` })
    );

    if (config.dropout) {
      model.add(
        tf.layers.dropout({
          rate: config.dropout,
          name: `dropout${layerIndex}`,
        })
      );
    }
  }

  // Flatten the output for the dense layers
  model.add(tf.layers.flatten({ name: "flatten" }));

  // Dense layers for feature extraction
  model.add(
    tf.layers.dense({ units: 512, activation: "relu", name: "dense1" })
  );
  model.add(tf.layers.dropout({ rate: 0.5, name: "dropout_dense" }));

  // Output layer - different number of classes based on image type
  const outputUnits = getOutputUnits(imageType);
  model.add(
    tf.layers.dense({
      units: outputUnits,
      activation: "softmax",
      name: "output",
    })
  );

  // Compile the model
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// Helper functions for model configuration based on image type
function getInitialFilters(imageType: string): number {
  switch (imageType) {
    case "mri":
      return 32;
    case "ct":
      return 32;
    case "xray":
      return 24;
    case "ultrasound":
      return 16;
    default:
      return 32;
  }
}

function getLayerConfig(
  imageType: string
): Array<{ filters: number; kernelSize: number; dropout?: number }> {
  switch (imageType) {
    case "mri":
      return [
        { filters: 64, kernelSize: 3 },
        { filters: 128, kernelSize: 3 },
        { filters: 256, kernelSize: 3, dropout: 0.3 },
      ];
    case "ct":
      return [
        { filters: 64, kernelSize: 3 },
        { filters: 128, kernelSize: 3 },
        { filters: 256, kernelSize: 3, dropout: 0.3 },
      ];
    case "xray":
      return [
        { filters: 48, kernelSize: 3 },
        { filters: 96, kernelSize: 3, dropout: 0.2 },
      ];
    case "ultrasound":
      return [
        { filters: 32, kernelSize: 3 },
        { filters: 64, kernelSize: 3, dropout: 0.2 },
      ];
    default:
      return [
        { filters: 64, kernelSize: 3 },
        { filters: 128, kernelSize: 3 },
      ];
  }
}

function getOutputUnits(imageType: string): number {
  switch (imageType) {
    case "mri":
      return 10;
    case "ct":
      return 8;
    case "xray":
      return 6;
    case "ultrasound":
      return 5;
    default:
      return 10;
  }
}

// Preprocess the image for the CNN model
async function preprocessImage(
  imageBuffer: ArrayBuffer,
  imageType: string
): Promise<tf.Tensor4D> {
  // Convert the image buffer to a tensor
  const image = await tf.browser.fromPixelsAsync(
    await createImageBitmap(new Blob([imageBuffer]))
  );

  // Resize to the expected input size (224x224)
  const resized = tf.image.resizeBilinear(image, [224, 224]);

  // Convert to grayscale for MRI and CT, keep color channels for others
  let processed;
  if (imageType === "mri" || imageType === "ct") {
    processed = resized.mean(2, true);
  } else {
    processed = resized;
  }

  // Normalize pixel values to [0, 1]
  const normalized = processed.div(tf.scalar(255));

  // Add batch dimension
  const batched = normalized.expandDims(0) as tf.Tensor4D;

  // Clean up intermediate tensors
  image.dispose();
  resized.dispose();
  if (processed !== resized) {
    processed.dispose();
  }

  return batched;
}

// Extract features from the CNN model
async function extractFeatures(
  preprocessedImage: tf.Tensor4D,
  imageType: string
): Promise<any> {
  const model = models[imageType];
  if (!model) {
    throw new Error(`Model for ${imageType} not loaded`);
  }

  // Get activations from an intermediate layer for feature extraction
  const featureExtractionModel = tf.model({
    inputs: model.inputs,
    outputs: model.getLayer("dense1").output,
  });

  // Get the feature vector
  const features = featureExtractionModel.predict(
    preprocessedImage
  ) as tf.Tensor;

  // Convert to a JavaScript array
  const featureArray = await features.array();

  // Clean up
  features.dispose();

  return featureArray;
}

// Run a forward pass through the model to get predictions
async function getPredictions(
  preprocessedImage: tf.Tensor4D,
  imageType: string
): Promise<any> {
  const model = models[imageType];
  if (!model) {
    throw new Error(`Model for ${imageType} not loaded`);
  }

  // Get model predictions
  const predictions = model.predict(preprocessedImage) as tf.Tensor;

  // Convert to a JavaScript array
  const predictionArray = await predictions.array();

  // Clean up
  predictions.dispose();

  return predictionArray;
}

// Main function to analyze a medical image with the CNN model
export async function analyzeWithCNN(
  imageBuffer: ArrayBuffer,
  imageType = "mri"
): Promise<any> {
  // Validate image type
  if (!Object.keys(models).includes(imageType)) {
    throw new Error(`Unsupported image type: ${imageType}`);
  }

  // Ensure the model is loaded
  await loadModel(imageType);

  // Preprocess the image
  const preprocessedImage = await preprocessImage(imageBuffer, imageType);

  try {
    // Extract features
    const features = await extractFeatures(preprocessedImage, imageType);

    // Get predictions
    const predictions = await getPredictions(preprocessedImage, imageType);

    // Get categories based on image type
    const categories = getCategoriesForImageType(imageType);

    // Simulate detection of regions of interest
    // In a real application, this would be based on actual model outputs
    const regionsOfInterest = simulateRegionsOfInterest(imageType);

    // Return a structured analysis result
    return {
      features: {
        summary: `Feature vector extracted from dense layer for ${imageType.toUpperCase()}`,
        dimensionality: features[0].length,
        // Include only a sample of the features to keep the response size reasonable
        sample: features[0].slice(0, 10),
      },
      predictions: {
        categories,
        confidenceScores: predictions[0],
      },
      regionsOfInterest,
      analysisMetadata: {
        modelArchitecture: `CNN optimized for ${imageType.toUpperCase()} analysis`,
        inputDimensions:
          imageType === "mri" || imageType === "ct" ? "224x224x1" : "224x224x3",
        processingTime: Math.floor(Math.random() * 500 + 500) + "ms", // Simulated processing time
        confidenceThreshold: 0.7,
        imageType: imageType.toUpperCase(),
      },
    };
  } finally {
    // Clean up tensors
    preprocessedImage.dispose();
    tf.disposeVariables();
  }
}

// Get categories based on image type
function getCategoriesForImageType(imageType: string): string[] {
  switch (imageType) {
    case "mri":
      return [
        "Normal tissue",
        "Brain tumor - Glioma",
        "Brain tumor - Meningioma",
        "Brain tumor - Pituitary",
        "Potential abnormality",
        "Inflammation",
        "Fluid accumulation",
        "Bone abnormality",
        "Tissue damage",
        "Vascular issue",
      ];
    case "ct":
      return [
        "Normal tissue",
        "Bone abnormality",
        "Soft tissue mass",
        "Fluid collection",
        "Calcification",
        "Hemorrhage",
        "Air/gas presence",
        "Foreign body",
      ];
    case "xray":
      return [
        "Normal",
        "Pneumonia",
        "Fracture",
        "Mass/nodule",
        "Pleural effusion",
        "Cardiomegaly",
      ];
    case "ultrasound":
      return [
        "Normal tissue",
        "Cyst",
        "Solid mass",
        "Fluid collection",
        "Calcification",
      ];
    default:
      return ["Normal", "Abnormal", "Requires further investigation"];
  }
}

// Simulate detection of regions of interest in the medical image
// In a real application, this would be based on actual model outputs
function simulateRegionsOfInterest(imageType: string) {
  // Generate 1-4 regions of interest
  const numRegions = Math.floor(Math.random() * 4) + 1;
  const regions = [];

  for (let i = 0; i < numRegions; i++) {
    regions.push({
      id: i + 1,
      location: {
        x: Math.floor(Math.random() * 224),
        y: Math.floor(Math.random() * 224),
        width: Math.floor(Math.random() * 50) + 20,
        height: Math.floor(Math.random() * 50) + 20,
      },
      confidence: (Math.random() * 0.3 + 0.7).toFixed(2), // 0.7-1.0
      classification: getRandomClassificationForImageType(imageType),
      intensity: (Math.random() * 100).toFixed(1),
    });
  }

  return regions;
}

// Get random classification based on image type
function getRandomClassificationForImageType(imageType: string): string {
  const classifications = {
    mri: [
      "Potential tumor region",
      "Abnormal tissue density",
      "Region of interest",
      "Signal intensity change",
      "Contrast enhancement",
      "Tumor boundary",
      "Edema surrounding lesion",
      "Glioma characteristics",
      "Tumor core",
      "Necrotic tissue",
    ],
    ct: [
      "Density variation",
      "Potential lesion",
      "Contrast enhancement",
      "Structural abnormality",
      "Tissue boundary",
    ],
    xray: [
      "Opacity",
      "Radiolucency",
      "Structural change",
      "Potential fracture",
      "Soft tissue mass",
    ],
    ultrasound: [
      "Hypoechoic region",
      "Hyperechoic region",
      "Cystic structure",
      "Solid mass",
      "Acoustic shadowing",
    ],
  };

  const options =
    classifications[imageType as keyof typeof classifications] ||
    classifications.mri;
  return options[Math.floor(Math.random() * options.length)];
}
