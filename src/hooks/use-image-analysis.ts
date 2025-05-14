"use client";

import { useState } from "react";

import { analyzeImage } from "@/lib/analyze-image";
import { useToast } from "./use-toast";

type AnalysisType = "general" | "detailed" | "medical";
type AnalysisStatus = "idle" | "loading" | "success" | "error";
type ImageType = "mri" | "ct" | "xray" | "ultrasound";

export function useImageAnalysis() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [analysisType, setAnalysisType] = useState<AnalysisType>("general");
  const [imageType, setImageType] = useState<ImageType>("mri");
  const [customPrompt, setCustomPrompt] = useState("");
  const [analysisResult, setAnalysisResult] = useState<string>("");
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [showRegions, setShowRegions] = useState(true);
  const [regionsOfInterest, setRegionsOfInterest] = useState<any[]>([]);
  const { toast } = useToast();

  const handleImageUpload = (file: File) => {
    // Check file size (3MB limit)
    const MAX_FILE_SIZE = 3 * 1024 * 1024; // 3MB in bytes
    if (file.size > MAX_FILE_SIZE) {
      toast({
        variant: "destructive",
        title: "File too large",
        description: "Please upload an image smaller than 3MB",
      });
      return;
    }

    if (file.type.startsWith("image/")) {
      // Set image type to MRI by default for brain images
      setImageType("mri");

      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      toast({
        variant: "destructive",
        title: "Invalid file type",
        description: "Please upload an image file",
      });
    }
  };

  const getPromptForAnalysisType = (
    type: AnalysisType,
    custom: string
  ): string => {
    if (custom.trim()) return custom;

    switch (type) {
      case "general":
        return `Analyze this ${imageType.toUpperCase()} image and provide a general overview of what can be observed.${
          imageType === "mri"
            ? " If this appears to be a brain tumor, please identify the type and characteristics."
            : ""
        }`;
      case "detailed":
        return `Provide a detailed analysis of this ${imageType.toUpperCase()} image, including potential abnormalities, structures visible, and any notable observations.${
          imageType === "mri"
            ? " If this shows a brain tumor, provide detailed assessment of tumor type, location, size, and effects on surrounding structures."
            : ""
        }`;
      case "medical":
        return `Based on this ${imageType.toUpperCase()} image, provide potential medical insights and recommendations for further investigation if needed.${
          imageType === "mri"
            ? " If a brain tumor is present, discuss potential diagnosis, differential considerations, and recommended follow-up procedures."
            : ""
        }`;
      default:
        return `Analyze this ${imageType.toUpperCase()} image.`;
    }
  };

  const handleAnalysis = async () => {
    if (!selectedImage) {
      toast({
        variant: "destructive",
        title: "No image selected",
        description: "Please upload a medical image first",
      });
      return;
    }

    setStatus("loading");
    try {
      const prompt = customPrompt || getPromptForAnalysisType(analysisType, "");
      const result = await analyzeImage(selectedImage, prompt, imageType);
      setAnalysisResult(result.analysis);
      setRegionsOfInterest(result.regions || []);
      setStatus("success");
      toast({
        title: "Analysis complete",
        description: `Your ${imageType.toUpperCase()} image has been analyzed successfully`,
      });
    } catch (error) {
      console.error("Analysis error:", error);
      setStatus("error");
      toast({
        variant: "destructive",
        title: "Analysis failed",
        description:
          error instanceof Error ? error.message : "An unknown error occurred",
      });
    }
  };

  return {
    selectedImage,
    imagePreview,
    analysisType,
    imageType,
    customPrompt,
    analysisResult,
    status,
    showRegions,
    regionsOfInterest,
    handleImageUpload,
    setAnalysisType,
    setImageType,
    setCustomPrompt,
    getPromptForAnalysisType,
    handleAnalysis,
    setShowRegions,
  };
}
