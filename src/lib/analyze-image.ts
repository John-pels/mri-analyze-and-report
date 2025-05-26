import { RegionOfInterest } from "@/types/medical-analysis";

interface ImageAnalysisResult {
  analysis: string;
  regions: RegionOfInterest[];
  error?: string;
  // Add other specific fields that your API returns
}

export async function analyzeImage(
  image: File,
  prompt: string,
  imageType = "mri"
): Promise<ImageAnalysisResult> {
  try {
    // Create a FormData object to send the image and prompt to the server
    const formData = new FormData();
    formData.append("image", image);
    formData.append("prompt", prompt);
    formData.append("imageType", imageType);

    // Call the server-side API route
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.error || `Server responded with status: ${response.status}`
      );
    }

    // Parse and return the analysis results
    const result = await response.json();
    return result as ImageAnalysisResult;
  } catch (error) {
    console.error("Error analyzing image:", error);
    throw new Error(
      `Failed to analyze the ${imageType.toUpperCase()} image. Please try again.`
    );
  }
}
