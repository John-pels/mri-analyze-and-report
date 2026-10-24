import { analyzeWithCNN, type ImageType } from "@/lib/cnn-model";
import type { CNNAnalysisResult, RegionOfInterest } from "@/types/medical-analysis";

interface ImageAnalysisResult {
  analysis: string;
  regions: RegionOfInterest[];
  cnn: CNNAnalysisResult;
  error?: string;
}

// 1. Classify the image in the browser (pixels stay on the device).
// 2. Send only the structured CNN result and the prompt to the server,
//    which asks an LLM to draft a report.
export async function analyzeImage(
  image: File,
  prompt: string,
  imageType: ImageType = "mri"
): Promise<ImageAnalysisResult> {
  const cnn = await analyzeWithCNN(image, imageType);

  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cnn, prompt, imageType }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Server responded with status: ${response.status}`);
  }

  const result = await response.json();
  return { analysis: result.analysis, regions: cnn.regionsOfInterest, cnn };
}
