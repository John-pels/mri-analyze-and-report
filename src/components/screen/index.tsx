"use client";

import { ImageViewer } from "@/components/image-viewer";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useDropzoneHandler } from "@/hooks/use-dropzone-handler";
import { useImageAnalysis } from "@/hooks/use-image-analysis";
import { useMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";
import {
  ArrowDown,
  Brain,
  BrainCircuit,
  CheckIcon,
  ClipboardList,
  CopyIcon,
  FileUp,
  FlaskConical,
  Layers,
  Loader2,
  Microscope,
  Scan,
  Stethoscope,
  UploadCloud,
} from "lucide-react";
import { type ChangeEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface AnalysisTypeValue {
  value: "general" | "detailed" | "medical";
}

const normalizeMarkdown = (text: string) => {
  return text.replace(/^\*(?!\*)/gm, "* ").replace(/\*\*(.*?)\*\*/g, "**$1**");
};

export function LandingPage() {
  const {
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
    selectedImage,
  } = useImageAnalysis();

  const { getRootProps, getInputProps, isDragActive } = useDropzoneHandler({
    onImageUpload: handleImageUpload,
  });

  const [copied, setCopied] = useState(false);
  const [hasUserEditedPrompt, setHasUserEditedPrompt] = useState(false);
  const isMobile = useMobile();
  const textareaRef = useRef(null);

  const handleCopy = async (analysisResult: string) => {
    try {
      await navigator.clipboard.writeText(analysisResult);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy", error);
    }
  };

  const handlePromptChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setCustomPrompt(e.target.value);
    setHasUserEditedPrompt(true);
  };

  // Only update the custom prompt when analysis type changes if user hasn't edited it
  useEffect(() => {
    if (!hasUserEditedPrompt) {
      setCustomPrompt(getPromptForAnalysisType(analysisType, ""));
    }
  }, [analysisType, getPromptForAnalysisType, setCustomPrompt, hasUserEditedPrompt]);

  // Reset the user edit flag when analysis type changes

  const handleAnalysisTypeChange = (value: AnalysisTypeValue["value"]): void => {
    setAnalysisType(value);
    setHasUserEditedPrompt(false);
  };

  // Scroll to results when analysis is complete on mobile
  useEffect(() => {
    if (status === "success" && isMobile) {
      setTimeout(() => {
        document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth" });
      }, 500);
    }
  }, [status, isMobile]);

  return (
    <main className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Medical Image Analysis Assistant</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Upload a medical image and get AI-powered analysis using advanced CNN and LLM technology
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-6">
          <Card
            className={cn(
              "border-2 border-dashed rounded-lg p-6 transition-colors",
              isDragActive ? "border-primary bg-primary/5" : "border-border",
            )}
          >
            <div
              {...getRootProps()}
              className="flex flex-col items-center justify-center h-64 cursor-pointer"
            >
              <input {...getInputProps()} />
              {imagePreview ? (
                <div className="relative w-full h-full">
                  <ImageViewer
                    src={imagePreview || "/placeholder.svg"}
                    alt={`${imageType.toUpperCase()} Preview`}
                    regions={showRegions ? regionsOfInterest : []}
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-4">
                  <UploadCloud className="h-12 w-12 text-muted-foreground mb-4 animate-pulse" />
                  <h3 className="font-medium text-lg mb-1">Drag & drop your medical image</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    or click to browse files (PNG, JPG, JPEG)
                  </p>
                  <Button variant="outline" size="sm">
                    <FileUp className="h-4 w-4 mr-2" />
                    Select File
                  </Button>
                </div>
              )}
            </div>
          </Card>

          <div className="space-y-2">
            <label htmlFor="image-type" className="text-sm font-medium">
              Image Type
            </label>
            <Select
              value={imageType}
              onValueChange={(value: "mri" | "ct" | "xray" | "ultrasound") => setImageType(value)}
            >
              <SelectTrigger id="image-type">
                <SelectValue placeholder="Select image type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mri">
                  <div className="flex items-center">
                    <Brain className="h-4 w-4 mr-2" />
                    MRI Scan
                  </div>
                </SelectItem>
                <SelectItem value="ct">
                  <div className="flex items-center">
                    <Layers className="h-4 w-4 mr-2" />
                    CT Scan
                  </div>
                </SelectItem>
                <SelectItem value="xray">
                  <div className="flex items-center">
                    <Scan className="h-4 w-4 mr-2" />
                    X-Ray
                  </div>
                </SelectItem>
                <SelectItem value="ultrasound">
                  <div className="flex items-center">
                    <Microscope className="h-4 w-4 mr-2" />
                    Ultrasound
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {status === "success" && regionsOfInterest.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="show-regions" className="text-sm font-medium">
                  Show Regions of Interest
                </Label>
                <Switch id="show-regions" checked={showRegions} onCheckedChange={setShowRegions} />
              </div>
            </div>
          )}

          <Tabs
            defaultValue="general"
            onValueChange={(value) => handleAnalysisTypeChange(value as AnalysisTypeValue["value"])}
          >
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="general">
                <Scan className="h-5 w-5 mr-2" />
                <span className="hidden md:inline md:mr-1">General</span>
                Analysis
              </TabsTrigger>
              <TabsTrigger value="detailed">
                <BrainCircuit className="h-5 w-5 mr-2" />
                <span className="hidden md:inline md:mr-1">Detailed</span>
                Report
              </TabsTrigger>
              <TabsTrigger value="medical">
                <ClipboardList className="h-5 w-5 mr-2" />
                <span className="hidden md:inline md:mr-1">Medical</span>
                Insights
              </TabsTrigger>
            </TabsList>

            <TabsContent value="general">
              <p className="text-sm text-muted-foreground mb-2">
                Get a general overview of the medical image with key observations.
              </p>
            </TabsContent>
            <TabsContent value="detailed">
              <p className="text-sm text-muted-foreground mb-2">
                Receive a comprehensive analysis with detailed findings and potential abnormalities.
              </p>
            </TabsContent>
            <TabsContent value="medical">
              <p className="text-sm text-muted-foreground mb-2">
                Get medical insights and recommendations based on the medical image.
              </p>
            </TabsContent>
          </Tabs>

          <div className="space-y-2">
            <label htmlFor="custom-prompt" className="text-sm font-medium">
              Custom Prompt (Optional)
            </label>
            <Textarea
              ref={textareaRef}
              id="custom-prompt"
              placeholder="Enter a custom prompt for the analysis..."
              value={customPrompt}
              onChange={handlePromptChange}
              disabled={status === "loading"}
              className="min-h-[96px] resize-y"
              rows={4}
            />
          </div>

          <Button
            type="button"
            onClick={handleAnalysis}
            disabled={!selectedImage || status === "loading"}
            className="w-full text-white"
          >
            {status === "loading" ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin text-white" />
                Analyzing...
              </>
            ) : (
              <>
                <FlaskConical className="h-5 w-5 mr-2 text-white" />
                Analyze Medical Image
              </>
            )}
          </Button>

          {/* Mobile indicator for results */}
          {status !== "success" && isMobile && (
            <div className="md:hidden mt-6 flex flex-col items-center justify-center animate-bounce">
              <p className="text-sm text-muted-foreground mb-2">Scroll down to see results</p>
              <ArrowDown className="h-5 w-5 text-primary" />
            </div>
          )}

          {/* Mobile view results button */}
          {status === "success" && isMobile && (
            <div className="md:hidden mt-4">
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => {
                  document
                    .getElementById("results-section")
                    ?.scrollIntoView({ behavior: "smooth" });
                }}
              >
                <ArrowDown className="h-4 w-4 mr-2" />
                View Analysis Results
              </Button>
            </div>
          )}
        </div>

        <div id="results-section">
          <Card className="p-6 h-full">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold flex items-center">
                <Stethoscope className="h-5 w-5 mr-2" />
                Analysis Results
              </h2>
              {status === "success" && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => handleCopy(analysisResult)}
                >
                  {copied ? <CheckIcon className="w-4 h-4" /> : <CopyIcon className="w-4 h-4" />}
                </Button>
              )}
            </div>

            <div className="bg-muted/40 rounded-lg p-4 h-[calc(100%-3rem)] overflow-auto">
              {status === "loading" ? (
                <div className="space-y-2 h-full flex flex-col justify-center">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-4/5" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-full" />
                </div>
              ) : status === "success" ? (
                <div>
                  {regionsOfInterest.length > 0 && (
                    <div className="mb-4 p-3 bg-muted rounded-md">
                      <h3 className="text-sm font-medium mb-2">Detected Regions of Interest</h3>
                      <div className="space-y-2">
                        {regionsOfInterest.map((region) => (
                          <div
                            key={`${region.confidence}-${region.location.y + region.location.x}`}
                            className="text-xs p-2 bg-background rounded border"
                          >
                            <div className="flex justify-between">
                              <span className="font-medium">{region.classification}</span>
                              <span className="text-muted-foreground">
                                Confidence: {region.confidence}
                              </span>
                            </div>
                            <div className="text-muted-foreground mt-1">
                              Location: x:{region.location.x}, y:{region.location.y}, w:
                              {region.location.width}, h:
                              {region.location.height}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    {analysisResult.split("\n").map((line) => (
                      <ReactMarkdown key={line} remarkPlugins={[remarkGfm]}>
                        {normalizeMarkdown(line)}
                      </ReactMarkdown>
                    ))}
                  </div>
                </div>
              ) : status === "error" ? (
                <div className="text-destructive text-center py-8">
                  <p>An error occurred during analysis. Please try again.</p>
                </div>
              ) : (
                <div className="text-muted-foreground text-center py-8">
                  <p>Upload a medical image and select an analysis type to get started.</p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </main>
  );
}
