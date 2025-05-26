export interface UsageMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  apiUsageCount: {
    gemini: number;
    openai: number;
    fallback: number;
  };
  averageResponseTime: number;
  requestsByHour: Record<string, number>;
  errorTypes: Record<string, number>;
  imageTypes: Record<string, number>;
  lastReset: Date;
}

export interface RequestLog {
  timestamp: Date;
  apiUsed: string;
  success: boolean;
  responseTime: number;
  imageType: string;
  fileSize: number;
  errorType?: string;
  userId?: string;
}

export interface CacheEntry {
  data: { metadata?: { fileSize?: number } };
  timestamp: Date;
  expiresAt: Date;
  accessCount: number;
  lastAccessed: Date;
}

export interface CacheOptions {
  ttl?: number; // Time to live in milliseconds
  maxSize?: number; // Maximum number of entries
  enableSimilarity?: boolean; // Enable similarity-based caching
}

export interface QueueJob {
  id: string;
  imageBuffer: ArrayBuffer;
  imageType: string;
  prompt?: string;
  priority: number;
  createdAt: Date;
  attempts: number;
  maxAttempts: number;
  resolve: (
    value:
      | { analysis: string; jobId: string; processedAt: string }
      | PromiseLike<{ analysis: string; jobId: string; processedAt: string }>
  ) => void;
  reject: (reason?: unknown) => unknown;
}

export interface QueueStats {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  totalProcessed: number;
  averageProcessingTime: number;
}

export interface RateLimitConfig {
  maxRequests: number;
  timeWindowMs: number;
  enableBurst?: boolean;
  burstLimit?: number;
}

export interface RateLimitResult {
  allowed: boolean;
  resetTime: number;
  remainingRequests: number;
  retryAfter?: number;
}

export interface FallbackOptions {
  includeDisclaimer?: boolean;
  customFindings?: string[];
  severity?: "low" | "medium" | "high";
}

export interface AnalysisTemplate {
  title: string;
  generalFindings: string[];
  recommendations: string[];
  specificChecks: string[];
  disclaimer: string;
}

export interface CNNAnalysisResult {
  features: {
    summary: string;
    dimensionality: number;
    sample: number[];
  };
  predictions: {
    categories: string[];
    confidenceScores: number[];
  };
  regionsOfInterest: RegionOfInterest[];
  analysisMetadata: {
    modelArchitecture: string;
    inputDimensions: string;
    processingTime: string;
    confidenceThreshold: number;
    imageType: string;
  };
}

export interface RegionOfInterest {
  id: number;
  location: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  classification: string;
  intensity: string;
}
