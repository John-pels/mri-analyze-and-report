import type { RequestLog, UsageMetrics } from "@/types/medical-analysis";

class UsageMonitor {
  private static instance: UsageMonitor;
  private metrics: UsageMetrics;
  private requestLogs: RequestLog[] = [];
  private readonly MAX_LOGS = 1000; // Keep last 1000 requests

  private constructor() {
    this.metrics = this.initializeMetrics();
  }

  static getInstance(): UsageMonitor {
    if (!UsageMonitor.instance) {
      UsageMonitor.instance = new UsageMonitor();
    }
    return UsageMonitor.instance;
  }

  private initializeMetrics(): UsageMetrics {
    return {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      apiUsageCount: {
        gemini: 0,
        openai: 0,
        fallback: 0,
      },
      averageResponseTime: 0,
      requestsByHour: {},
      errorTypes: {},
      imageTypes: {},
      lastReset: new Date(),
    };
  }

  logRequest(requestData: {
    apiUsed: string;
    success: boolean;
    responseTime: number;
    imageType: string;
    fileSize: number;
    errorType?: string;
    userId?: string;
  }) {
    const log: RequestLog = {
      timestamp: new Date(),
      ...requestData,
    };

    // Add to request logs
    this.requestLogs.push(log);
    if (this.requestLogs.length > this.MAX_LOGS) {
      this.requestLogs.shift(); // Remove oldest log
    }

    // Update metrics
    this.updateMetrics(log);

    // Log to console for development
    if (process.env.NODE_ENV === "development") {
      console.log("[Usage Monitor]", {
        api: log.apiUsed,
        success: log.success,
        responseTime: `${log.responseTime}ms`,
        imageType: log.imageType,
        fileSize: `${Math.round(log.fileSize / 1024)}KB`,
      });
    }
  }

  private updateMetrics(log: RequestLog) {
    this.metrics.totalRequests++;

    if (log.success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
      if (log.errorType) {
        this.metrics.errorTypes[log.errorType] = (this.metrics.errorTypes[log.errorType] || 0) + 1;
      }
    }

    // Update API usage count
    if (log.apiUsed in this.metrics.apiUsageCount) {
      this.metrics.apiUsageCount[log.apiUsed as keyof typeof this.metrics.apiUsageCount]++;
    }

    // Update image type count
    this.metrics.imageTypes[log.imageType] = (this.metrics.imageTypes[log.imageType] || 0) + 1;

    // Update hourly requests
    const hour = log.timestamp.toISOString().slice(0, 13); // YYYY-MM-DDTHH
    this.metrics.requestsByHour[hour] = (this.metrics.requestsByHour[hour] || 0) + 1;

    // Update average response time
    const totalResponseTime = this.requestLogs
      .filter((l) => l.success)
      .reduce((sum, l) => sum + l.responseTime, 0);
    const successfulCount = this.requestLogs.filter((l) => l.success).length;

    this.metrics.averageResponseTime =
      successfulCount > 0 ? totalResponseTime / successfulCount : 0;
  }

  getMetrics(): UsageMetrics {
    return { ...this.metrics };
  }

  getRecentLogs(limit = 50): RequestLog[] {
    return this.requestLogs
      .slice(-limit)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  resetMetrics() {
    this.metrics = this.initializeMetrics();
    this.requestLogs = [];
    console.log("[Usage Monitor] Metrics reset");
  }

  // Get usage patterns for optimization
  getUsagePatterns() {
    const patterns = {
      peakHours: this.getPeakHours(),
      mostUsedAPI: this.getMostUsedAPI(),
      commonImageTypes: this.getCommonImageTypes(),
      errorRate: this.getErrorRate(),
      averageFileSize: this.getAverageFileSize(),
    };

    return patterns;
  }

  private getPeakHours(): string[] {
    const hourCounts = Object.entries(this.metrics.requestsByHour)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([hour]) => hour);

    return hourCounts;
  }

  private getMostUsedAPI(): string {
    const apiCounts = this.metrics.apiUsageCount;
    return Object.entries(apiCounts).sort(([, a], [, b]) => b - a)[0]?.[0] || "none";
  }

  private getCommonImageTypes(): string[] {
    return Object.entries(this.metrics.imageTypes)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([type]) => type);
  }

  private getErrorRate(): number {
    if (this.metrics.totalRequests === 0) return 0;
    return (this.metrics.failedRequests / this.metrics.totalRequests) * 100;
  }

  private getAverageFileSize(): number {
    if (this.requestLogs.length === 0) return 0;
    const totalSize = this.requestLogs.reduce((sum, log) => sum + log.fileSize, 0);
    return totalSize / this.requestLogs.length;
  }
}

export const usageMonitor = UsageMonitor.getInstance();
export type { UsageMetrics, RequestLog };
