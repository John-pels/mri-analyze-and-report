import type { RateLimitConfig, RateLimitResult } from "@/types/medical-analysis";

class RateLimiter {
  private requests: number[] = [];
  private burstRequests: number[] = [];
  private readonly config: Required<RateLimitConfig>;

  constructor(config: RateLimitConfig) {
    this.config = {
      enableBurst: false,
      burstLimit: config.maxRequests * 2,
      ...config,
    };
  }

  checkLimit(): RateLimitResult {
    const now = Date.now();

    // Clean old requests
    this.requests = this.requests.filter((time) => now - time < this.config.timeWindowMs);

    // Clean old burst requests if burst is enabled
    if (this.config.enableBurst) {
      this.burstRequests = this.burstRequests.filter(
        (time) => now - time < 60000, // 1 minute burst window
      );
    }

    const currentRequests = this.requests.length;
    const currentBurstRequests = this.burstRequests.length;

    // Check burst limit first if enabled
    if (this.config.enableBurst && currentBurstRequests >= this.config.burstLimit) {
      return {
        allowed: false,
        resetTime: this.getResetTime(),
        remainingRequests: 0,
        retryAfter: Math.ceil((this.burstRequests[0] + 60000 - now) / 1000),
      };
    }

    // Check regular limit
    if (currentRequests >= this.config.maxRequests) {
      return {
        allowed: false,
        resetTime: this.getResetTime(),
        remainingRequests: 0,
        retryAfter: Math.ceil((this.requests[0] + this.config.timeWindowMs - now) / 1000),
      };
    }

    // Allow request and record it
    this.requests.push(now);
    if (this.config.enableBurst) {
      this.burstRequests.push(now);
    }

    return {
      allowed: true,
      resetTime: this.getResetTime(),
      remainingRequests: this.config.maxRequests - this.requests.length,
    };
  }

  private getResetTime(): number {
    if (this.requests.length === 0) return 0;
    return this.requests[0] + this.config.timeWindowMs;
  }

  getStatus() {
    const now = Date.now();

    // Clean old requests for accurate status
    this.requests = this.requests.filter((time) => now - time < this.config.timeWindowMs);

    return {
      currentRequests: this.requests.length,
      maxRequests: this.config.maxRequests,
      timeWindowMs: this.config.timeWindowMs,
      resetTime: this.getResetTime(),
      remainingRequests: Math.max(0, this.config.maxRequests - this.requests.length),
      burstEnabled: this.config.enableBurst,
      burstLimit: this.config.burstLimit,
      currentBurstRequests: this.config.enableBurst
        ? this.burstRequests.filter((time) => now - time < 60000).length
        : 0,
    };
  }

  reset(): void {
    this.requests = [];
    this.burstRequests = [];
  }
}

// Global rate limiter instances
const globalRateLimiter = new RateLimiter({
  maxRequests: 10,
  timeWindowMs: 60000, // 1 minute
  enableBurst: true,
  burstLimit: 15,
});

// Per-IP rate limiter map
const ipRateLimiters = new Map<string, RateLimiter>();

// Clean up old IP rate limiters periodically
setInterval(() => {
  const now = Date.now();
  for (const [ip, limiter] of ipRateLimiters.entries()) {
    const status = limiter.getStatus();
    // Remove if no requests in the last hour
    if (status.currentRequests === 0 && now - status.resetTime > 3600000) {
      ipRateLimiters.delete(ip);
    }
  }
}, 300000); // Clean every 5 minutes

export function checkGlobalRateLimit(): RateLimitResult {
  return globalRateLimiter.checkLimit();
}

export function checkIPRateLimit(ip: string): RateLimitResult {
  if (!ipRateLimiters.has(ip)) {
    ipRateLimiters.set(
      ip,
      new RateLimiter({
        maxRequests: 20, // More lenient per-IP limit
        timeWindowMs: 60000,
        enableBurst: true,
        burstLimit: 25,
      }),
    );
  }

  const limiter = ipRateLimiters.get(ip)!;
  return limiter.checkLimit();
}

export function getRateLimitStatus() {
  return {
    global: globalRateLimiter.getStatus(),
    activeIPs: ipRateLimiters.size,
    ipLimits: Array.from(ipRateLimiters.entries()).map(([ip, limiter]) => ({
      ip: ip.replace(/\d+\.\d+\.\d+\./, "xxx.xxx.xxx."), // Anonymize IP
      status: limiter.getStatus(),
    })),
  };
}

export { RateLimiter };
export type { RateLimitConfig, RateLimitResult };
