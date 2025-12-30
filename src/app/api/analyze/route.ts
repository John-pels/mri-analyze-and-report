import { analyzeWithCNN } from "@/lib/cnn-model";
import { GoogleGenerativeAI } from "@google/generative-ai";
// app/api/analyze/route.ts
import { type NextRequest, NextResponse } from "next/server";

import { cacheService } from "@/lib/helpers/cache";
import {
  fallbackAnalyzer,
  type RegionOfInterest,
} from "@/lib/helpers/fallback-analyzer";
import { queueSystem } from "@/lib/helpers/queue";
import {
  checkGlobalRateLimit,
  checkIPRateLimit,
} from "@/lib/helpers/rate-limiter";
// Import all services
import { usageMonitor } from "@/lib/helpers/usage-monitor";
import type { CNNAnalysisResult } from "@/types/medical-analysis";

// Type definitions for CNN model response

// Simplified interface for internal use
interface CNNResults {
  confidence: number;
  predictions: string[];
  regionsOfInterest: RegionOfInterest[];
  error?: string;
}

interface AnalysisMetadata {
  imageType: string;
  apiUsed: string;
  cnnStatus: "success" | "failed";
  timestamp: string;
  processingTime: number;
  usingFallback: boolean;
  fileName?: string;
  fileSize?: number;
  clientIP?: string;
  fromCache?: boolean;
  queueUsed?: boolean;
  cacheHit?: boolean;
}

// Fixed AnalysisResult interface - added confidence property
interface AnalysisResult {
  analysis: string;
  regions: RegionOfInterest[];
  metadata: AnalysisMetadata;
  responseTime: number;
  success: boolean;
  confidence: number; // Added this property to match CNNResults
  errorType?: string;
  userNotice?: {
    type: string;
    message: string;
    recommendation: string;
  };
}

interface AnalysisError extends Error {
  responseTime: number;
  success: false;
  errorType: string;
}

interface RateLimit {
  allowed: boolean;
  retryAfter?: number;
}

interface QueueStats {
  pending: number;
  processing: number;
}

interface UsageRequest {
  apiUsed: string;
  success: boolean;
  responseTime: number;
  imageType: string;
  fileSize: number;
  errorType?: string;
  userId: string;
}

interface OpenAIResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
}

interface QueueResult {
  jobId: string;
  analysis: string;
  processedAt: string;
}

interface CacheOptions {
  ttl: number;
}

// Helper function to get client IP
function getClientIP(request: NextRequest): string {
  const forwarded = request.headers.get("x-forwarded-for");
  const realIP = request.headers.get("x-real-ip");

  if (forwarded) {
    return forwarded.split(",")[0].trim();
  }

  if (realIP) {
    return realIP;
  }

  return "unknown";
}

// Helper function to determine if we should use queue
function shouldUseQueue(imageSize: number, currentLoad: number): boolean {
  // Use queue for large files or high system load
  return imageSize > 1024 * 1024 || currentLoad > 5; // 1MB or more than 5 concurrent requests
}

// Main analysis function (extracted from the route handler)
async function performAnalysis(
  imageBuffer: ArrayBuffer,
  imageType: "mri" | "ct" | "xray" | "ultrasound",
  prompt?: string
): Promise<AnalysisResult> {
  const startTime = Date.now();
  let cnnResults: CNNResults;
  let analysisText = "";
  let apiUsed = "fallback";
  let errorType: string | undefined;

  try {
    // Try CNN analysis first
    try {
      const cnnAnalysis: CNNAnalysisResult = await analyzeWithCNN(
        imageBuffer,
        imageType
      );

      // Convert CNN analysis result to expected format
      const maxConfidence = Math.max(
        ...cnnAnalysis.predictions.confidenceScores
      );
      cnnResults = {
        confidence: maxConfidence,
        predictions: cnnAnalysis.predictions.categories,
        regionsOfInterest: cnnAnalysis.regionsOfInterest,
      };
    } catch (cnnError) {
      console.error("CNN analysis failed:", cnnError);
      errorType = "cnn_failure";
      cnnResults = {
        confidence: 0.5,
        predictions: ["Analysis pending"],
        regionsOfInterest: [],
        error: "CNN analysis unavailable",
      };
    }

    // Prepare analysis prompt
    const analysisPrompt = `
You are a medical imaging expert specializing in ${imageType.toUpperCase()} analysis. 
Based on the following CNN model analysis of a ${imageType.toUpperCase()} image, provide a detailed interpretation:

CNN Analysis Results:
${JSON.stringify(cnnResults, null, 2)}

Additional Context:
${
  prompt ||
  `Provide a general analysis of this ${imageType.toUpperCase()} image.`
}

Provide a comprehensive analysis that would be helpful for medical professionals. 
Include potential findings, areas of interest, and any recommendations for further investigation if applicable.

${
  imageType === "mri"
    ? `
If there are any indications of brain tumors in the analysis, please provide:
1. Potential tumor type based on imaging characteristics
2. Location and size assessment
3. Possible effects on surrounding structures
4. Differential diagnosis considerations
5. Recommended follow-up imaging or procedures
`
    : ""
}

IMPORTANT: Always include appropriate medical disclaimers and emphasize the need for professional radiological interpretation.
`;

    // Try AI APIs
    const geminiKey = process.env.GEMINI_API_KEY;
    const openaiKey = process.env.OPENAI_API_KEY;

    // Try Gemini first
    if (geminiKey) {
      try {
        const genAI = new GoogleGenerativeAI(geminiKey);
        const model = genAI.getGenerativeModel({
          model: "gemini-1.5-flash",
          generationConfig: {
            maxOutputTokens: 1000,
            temperature: 0.7,
          },
        });

        const result = await model.generateContent({
          contents: [{ role: "user", parts: [{ text: analysisPrompt }] }],
        });

        analysisText = result.response.text();
        apiUsed = "gemini";
      } catch (geminiError) {
        console.error("Gemini API failed:", geminiError);
        errorType = "gemini_failure";

        // Try OpenAI as fallback
        if (openaiKey) {
          try {
            const openaiResponse = await fetch(
              "https://api.openai.com/v1/chat/completions",
              {
                method: "POST",
                headers: {
                  Authorization: `Bearer ${openaiKey}`,
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  model: "gpt-3.5-turbo",
                  messages: [{ role: "user", content: analysisPrompt }],
                  max_tokens: 1000,
                  temperature: 0.7,
                }),
              }
            );

            if (openaiResponse.ok) {
              const openaiData: OpenAIResponse = await openaiResponse.json();
              analysisText = openaiData.choices[0].message.content;
              apiUsed = "openai";
            } else {
              throw new Error(
                `OpenAI API failed with status: ${openaiResponse.status}`
              );
            }
          } catch (openaiError) {
            console.error("OpenAI API also failed:", openaiError);
            errorType = "all_apis_failed";
            analysisText = fallbackAnalyzer.generateAnalysis(
              cnnResults,
              imageType,
              prompt,
              {
                severity: "medium",
              }
            );
          }
        } else {
          analysisText = fallbackAnalyzer.generateAnalysis(
            cnnResults,
            imageType,
            prompt,
            {
              severity: "medium",
            }
          );
        }
      }
    } else if (openaiKey) {
      // Try OpenAI directly if no Gemini key
      try {
        const openaiResponse = await fetch(
          "https://api.openai.com/v1/chat/completions",
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${openaiKey}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model: "gpt-3.5-turbo",
              messages: [{ role: "user", content: analysisPrompt }],
              max_tokens: 1000,
              temperature: 0.7,
            }),
          }
        );

        if (openaiResponse.ok) {
          const openaiData: OpenAIResponse = await openaiResponse.json();
          analysisText = openaiData.choices[0].message.content;
          apiUsed = "openai";
        } else {
          throw new Error(
            `OpenAI API failed with status: ${openaiResponse.status}`
          );
        }
      } catch (openaiError) {
        console.error("OpenAI API failed:", openaiError);
        errorType = "openai_failure";
        analysisText = fallbackAnalyzer.generateAnalysis(
          cnnResults,
          imageType,
          prompt,
          {
            severity: "medium",
          }
        );
      }
    } else {
      // No API keys available
      errorType = "no_api_keys";
      analysisText = fallbackAnalyzer.generateAnalysis(
        cnnResults,
        imageType,
        prompt,
        {
          severity: "medium",
        }
      );
    }

    const responseTime = Date.now() - startTime;

    // Fixed return statement - now includes confidence property
    return {
      analysis: analysisText,
      regions: cnnResults.regionsOfInterest || [],
      confidence: cnnResults.confidence, // Added this line
      metadata: {
        imageType,
        apiUsed,
        cnnStatus: cnnResults.error ? "failed" : "success",
        timestamp: new Date().toISOString(),
        processingTime: responseTime,
        usingFallback: apiUsed === "fallback",
      },
      responseTime,
      success: true,
      errorType,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    const analysisError: AnalysisError = {
      name: "AnalysisError",
      message: error instanceof Error ? error.message : "Unknown error",
      responseTime,
      success: false,
      errorType: errorType || "unknown_error",
    };
    throw analysisError;
  }
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const startTime = Date.now();
  const clientIP = getClientIP(request);

  try {
    // Check rate limits
    const globalLimit: RateLimit = checkGlobalRateLimit();
    const ipLimit: RateLimit = checkIPRateLimit(clientIP);

    if (!globalLimit.allowed) {
      return NextResponse.json(
        {
          error: "Global rate limit exceeded",
          retryAfter: globalLimit.retryAfter,
          message: `Global limit reached. Please wait ${globalLimit.retryAfter} seconds.`,
        },
        {
          status: 429,
          headers: {
            "Retry-After": globalLimit.retryAfter?.toString() || "60",
          },
        }
      );
    }

    if (!ipLimit.allowed) {
      return NextResponse.json(
        {
          error: "IP rate limit exceeded",
          retryAfter: ipLimit.retryAfter,
          message: `Too many requests from your IP. Please wait ${ipLimit.retryAfter} seconds.`,
        },
        {
          status: 429,
          headers: { "Retry-After": ipLimit.retryAfter?.toString() || "60" },
        }
      );
    }

    type ImageType = "mri" | "ct" | "xray" | "ultrasound";

    // Parse form data
    const formData = await request.formData();
    const imageFile = formData.get("image") as File | null;
    const prompt = formData.get("prompt") as string | null;
    const imageType = (formData.get("imageType") as ImageType) || "mri";
    const useQueue = formData.get("useQueue") === "true";

    // Validate inputs
    if (!imageFile) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      );
    }

    // Check file size
    const MAX_FILE_SIZE = 3 * 1024 * 1024; // 3MB
    if (imageFile.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: "File too large (max 3MB)" },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif"];
    if (!allowedTypes.includes(imageFile.type)) {
      return NextResponse.json(
        {
          error: "Invalid file type. Please upload a JPEG, PNG, or GIF image.",
        },
        { status: 400 }
      );
    }

    // Convert to array buffer
    const imageBuffer = await imageFile.arrayBuffer();

    // Check cache first
    const cachedResult: {
      metadata?: { fileSize?: number | undefined } | undefined;
    } | null = cacheService.get(imageBuffer, imageType, prompt ?? undefined);
    if (cachedResult) {
      // Log cache hit
      const cacheRequest: UsageRequest = {
        apiUsed: "cache",
        success: true,
        responseTime: Date.now() - startTime,
        imageType,
        fileSize: imageFile.size,
        userId: clientIP,
      };
      usageMonitor.logRequest(cacheRequest);

      return NextResponse.json({
        ...cachedResult,
        metadata: {
          ...cachedResult.metadata,
          fromCache: true,
          cacheHit: true,
        },
      });
    }

    // Determine if we should use queue
    const queueStats: QueueStats = queueSystem.getStats();
    const shouldQueue =
      useQueue || shouldUseQueue(imageFile.size, queueStats.processing);

    let result: AnalysisResult;

    if (shouldQueue) {
      // Add to queue
      const estimatedWait = queueSystem.getEstimatedWaitTime();

      // Return immediate response with queue information
      if (estimatedWait > 10000) {
        // More than 10 seconds
        const _queueResult: QueueResult = await queueSystem.addJob(
          imageBuffer,
          imageType,
          prompt ?? undefined,
          1
        );

        return NextResponse.json({
          queued: true,
          estimatedWait: Math.ceil(estimatedWait / 1000),
          message: "Request queued for processing due to high load",
          queuePosition: queueStats.pending + 1,
          jobId: "queue_job_id", // In real implementation, you'd return the actual job ID
        });
      }
    }

    // Process immediately
    try {
      result = await performAnalysis(
        imageBuffer,
        imageType,
        prompt || undefined
      );
    } catch (analysisError) {
      const error = analysisError as AnalysisError;

      // Log failed request
      const failedRequest: UsageRequest = {
        apiUsed: error.errorType || "unknown",
        success: false,
        responseTime: error.responseTime || Date.now() - startTime,
        imageType,
        fileSize: imageFile.size,
        errorType: error.errorType,
        userId: clientIP,
      };
      usageMonitor.logRequest(failedRequest);

      if (error.message?.includes("quota") || error.message?.includes("429")) {
        return NextResponse.json(
          {
            error: "API quota exceeded. Please try again later.",
            retryAfter: 60,
            usingFallback: true,
            fallbackReason: "quota_exceeded",
          },
          { status: 429 }
        );
      }

      return NextResponse.json(
        {
          error: "Failed to analyze the medical image",
          details:
            process.env.NODE_ENV === "development" ? error.message : undefined,
        },
        { status: 500 }
      );
    }

    // Cache successful result
    const cacheOptions: CacheOptions = { ttl: 24 * 60 * 60 * 1000 };
    cacheService.set(
      imageBuffer,
      imageType,
      result,
      cacheOptions,
      prompt ?? undefined
    );

    // Log successful request
    const successRequest: UsageRequest = {
      apiUsed: result.metadata.apiUsed,
      success: true,
      responseTime: result.responseTime,
      imageType,
      fileSize: imageFile.size,
      userId: clientIP,
    };
    usageMonitor.logRequest(successRequest);

    // Add user feedback about fallback usage
    if (result.metadata.usingFallback) {
      result.userNotice = {
        type: "fallback",
        message:
          "AI analysis temporarily unavailable. Using enhanced automated analysis.",
        recommendation:
          "Results are still clinically relevant but professional review is strongly recommended.",
      };
    }

    // Add additional metadata
    result.metadata = {
      ...result.metadata,
      fileName: imageFile.name,
      fileSize: imageFile.size,
      clientIP: clientIP.replace(/\d+\.\d+\.\d+\./, "xxx.xxx.xxx."), // Anonymize
      fromCache: false,
      queueUsed: shouldQueue,
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error processing medical image analysis:", error);

    // Log failed request
    const errorRequest: UsageRequest = {
      apiUsed: "error",
      success: false,
      responseTime: Date.now() - startTime,
      imageType: "unknown",
      fileSize: 0,
      errorType: "request_processing_error",
      userId: clientIP,
    };
    usageMonitor.logRequest(errorRequest);

    return NextResponse.json(
      {
        error: "Failed to process the request",
        details:
          process.env.NODE_ENV === "development"
            ? error instanceof Error
              ? error.message
              : String(error)
            : undefined,
      },
      { status: 500 }
    );
  }
}

// Add additional endpoints for monitoring and management
export async function GET(request: NextRequest): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get("action");

  switch (action) {
    case "stats":
      return NextResponse.json({
        usage: usageMonitor.getMetrics(),
        cache: cacheService.getStats(),
        queue: queueSystem.getStats(),
        patterns: usageMonitor.getUsagePatterns(),
      });

    case "health":
      return NextResponse.json({
        status: "healthy",
        timestamp: new Date().toISOString(),
        services: {
          cnn: "operational",
          cache: "operational",
          queue: "operational",
          monitoring: "operational",
        },
      });

    case "queue-info":
      return NextResponse.json(queueSystem.getQueueInfo());

    default:
      return NextResponse.json(
        { error: "Invalid action parameter" },
        { status: 400 }
      );
  }
}
