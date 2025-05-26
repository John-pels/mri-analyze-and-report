import crypto from "node:crypto";
import type { CacheEntry, CacheOptions } from "@/types/medical-analysis";

class CacheService {
  private static instance: CacheService;
  private cache: Map<string, CacheEntry> = new Map();
  private readonly defaultTTL = 24 * 60 * 60 * 1000; // 24 hours
  private readonly maxSize = 100; // Maximum cache entries
  private readonly similarityThreshold = 0.95; // 95% similarity for cache hit

  private constructor() {}

  static getInstance(): CacheService {
    if (!CacheService.instance) {
      CacheService.instance = new CacheService();
    }
    return CacheService.instance;
  }

  // Generate cache key from image buffer and analysis parameters
  private generateCacheKey(
    imageBuffer: ArrayBuffer,
    imageType: string,
    prompt?: string
  ): string {
    const hash = crypto.createHash("sha256");
    hash.update(new Uint8Array(imageBuffer));
    hash.update(imageType);
    if (prompt) {
      hash.update(prompt);
    }
    return hash.digest("hex");
  }

  // Generate similarity hash for approximate matching
  private generateSimilarityHash(imageBuffer: ArrayBuffer): string {
    // Simple similarity hash - in production, you might want to use perceptual hashing
    const sampleSize = Math.min(1024, imageBuffer.byteLength); // Sample first 1KB
    const sample = new Uint8Array(imageBuffer, 0, sampleSize);

    const hash = crypto.createHash("md5");
    // Reduce precision for similarity matching
    for (let i = 0; i < sample.length; i += 4) {
      hash.update(String(Math.floor(sample[i] / 16))); // Reduce to 16 levels
    }
    return hash.digest("hex");
  }

  // Store analysis result in cache
  set(
    imageBuffer: ArrayBuffer,
    imageType: string,
    data: { metadata?: { fileSize?: number } },
    options: CacheOptions = {},
    prompt?: string
  ): void {
    const key = this.generateCacheKey(imageBuffer, imageType, prompt);
    const ttl = options.ttl || this.defaultTTL;
    const maxSize = options.maxSize || this.maxSize;

    // Clean expired entries and enforce size limit
    this.cleanup();
    if (this.cache.size >= maxSize) {
      this.evictLeastUsed();
    }

    const entry: CacheEntry = {
      data,
      timestamp: new Date(),
      expiresAt: new Date(Date.now() + ttl),
      accessCount: 0,
      lastAccessed: new Date(),
    };

    this.cache.set(key, entry);

    if (process.env.NODE_ENV === "development") {
      console.log("[Cache] Stored entry:", {
        key: key.slice(0, 8),
        size: this.cache.size,
      });
    }
  }

  // Retrieve analysis result from cache
  get(
    imageBuffer: ArrayBuffer,
    imageType: string,
    prompt?: string,
    enableSimilarity = true
  ): { metadata?: { fileSize?: number } } | null {
    const key = this.generateCacheKey(imageBuffer, imageType, prompt);

    // Try exact match first
    let entry: CacheEntry | null | undefined = this.cache.get(key);

    if (!entry && enableSimilarity) {
      // Try similarity-based matching
      entry = this.findSimilarEntry(imageBuffer, imageType, prompt);
    }

    if (!entry) return null;

    // Check if expired
    if (entry.expiresAt < new Date()) {
      this.cache.delete(key);
      return null;
    }

    // Update access statistics
    entry.accessCount++;
    entry.lastAccessed = new Date();

    if (process.env.NODE_ENV === "development") {
      console.log("[Cache] Retrieved entry:", {
        key: key.slice(0, 8),
        accessCount: entry.accessCount,
        age: Date.now() - entry.timestamp.getTime(),
      });
    }

    return entry.data;
  }

  // Find similar cached entries
  private findSimilarEntry(
    imageBuffer: ArrayBuffer,
    _imageType: string,
    _prompt?: string
  ): CacheEntry | null {
    const _similarityHash = this.generateSimilarityHash(imageBuffer);

    for (const [_cachedKey, entry] of this.cache.entries()) {
      if (entry.expiresAt < new Date()) continue;

      // Extract original parameters from cached entries (this would need to be stored separately in production)
      // For now, we'll do a simple size-based similarity check
      const sizeDiff = Math.abs(
        imageBuffer.byteLength - (entry.data.metadata?.fileSize || 0)
      );
      const maxSize = Math.max(
        imageBuffer.byteLength,
        entry.data.metadata?.fileSize || 0
      );

      if (maxSize > 0) {
        const similarity = 1 - sizeDiff / maxSize;
        if (similarity >= this.similarityThreshold) {
          return entry;
        }
      }
    }

    return null;
  }

  // Clean up expired entries
  private cleanup(): void {
    const now = new Date();
    const keysToDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (entry.expiresAt < now) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      this.cache.delete(key);
    }

    if (keysToDelete.length > 0 && process.env.NODE_ENV === "development") {
      console.log(`[Cache] Cleaned up ${keysToDelete.length} expired entries`);
    }
  }

  // Evict least recently used entry
  private evictLeastUsed(): void {
    let lruKey: string | null = null;
    let oldestAccess = new Date();

    for (const [key, entry] of this.cache.entries()) {
      if (entry.lastAccessed < oldestAccess) {
        oldestAccess = entry.lastAccessed;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
      if (process.env.NODE_ENV === "development") {
        console.log("[Cache] Evicted LRU entry:", lruKey.slice(0, 8));
      }
    }
  }

  // Get cache statistics
  getStats() {
    this.cleanup(); // Clean before getting stats

    const entries = Array.from(this.cache.values());
    const now = Date.now();

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitRate: this.calculateHitRate(),
      averageAge:
        entries.length > 0
          ? entries.reduce(
              (sum, entry) => sum + (now - entry.timestamp.getTime()),
              0
            ) / entries.length
          : 0,
      totalAccesses: entries.reduce((sum, entry) => sum + entry.accessCount, 0),
      oldestEntry:
        entries.length > 0
          ? Math.min(...entries.map((entry) => entry.timestamp.getTime()))
          : null,
    };
  }

  private calculateHitRate(): number {
    // This would need to be tracked separately for accurate hit rate calculation
    // For now, return estimated hit rate based on access counts
    const entries = Array.from(this.cache.values());
    if (entries.length === 0) return 0;

    const totalAccesses = entries.reduce(
      (sum, entry) => sum + entry.accessCount,
      0
    );
    return totalAccesses > entries.length
      ? ((totalAccesses - entries.length) / totalAccesses) * 100
      : 0;
  }

  // Clear all cache entries
  clear(): void {
    const size = this.cache.size;
    this.cache.clear();
    console.log(`[Cache] Cleared ${size} entries`);
  }

  // Check if an entry exists (without accessing it)
  has(imageBuffer: ArrayBuffer, imageType: string, prompt?: string): boolean {
    const key = this.generateCacheKey(imageBuffer, imageType, prompt);
    const entry = this.cache.get(key);

    if (!entry) return false;
    if (entry.expiresAt < new Date()) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }
}

export const cacheService = CacheService.getInstance();
export type { CacheOptions, CacheEntry };
