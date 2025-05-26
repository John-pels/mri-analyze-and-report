import type { QueueJob, QueueStats } from "@/types/medical-analysis";

class QueueSystem {
  private static instance: QueueSystem;
  private queue: QueueJob[] = [];
  private processing: Map<string, QueueJob> = new Map();
  private maxConcurrent = 3;
  private currentlyProcessing = 0;
  private stats: QueueStats = {
    pending: 0,
    processing: 0,
    completed: 0,
    failed: 0,
    totalProcessed: 0,
    averageProcessingTime: 0,
  };
  private processingTimes: number[] = [];

  private constructor() {
    // Start the queue processor
    this.startProcessor();
  }

  static getInstance(): QueueSystem {
    if (!QueueSystem.instance) {
      QueueSystem.instance = new QueueSystem();
    }
    return QueueSystem.instance;
  }

  // Add job to queue
  async addJob(
    imageBuffer: ArrayBuffer,
    imageType: string,
    prompt?: string,
    priority = 1,
  ): Promise<{
    analysis: string;
    jobId: string;
    processedAt: string;
  }> {
    return new Promise((resolve, reject) => {
      const job: QueueJob = {
        id: this.generateJobId(),
        imageBuffer,
        imageType,
        prompt,
        priority,
        createdAt: new Date(),
        attempts: 0,
        maxAttempts: 3,
        resolve,
        reject,
      };

      // Insert job based on priority (higher priority = processed first)
      const insertIndex = this.queue.findIndex((queuedJob) => queuedJob.priority < priority);
      if (insertIndex === -1) {
        this.queue.push(job);
      } else {
        this.queue.splice(insertIndex, 0, job);
      }

      this.updateStats();

      if (process.env.NODE_ENV === "development") {
        console.log(
          `[Queue] Added job ${job.id} (priority: ${priority}, queue size: ${this.queue.length})`,
        );
      }
    });
  }

  private generateJobId(): string {
    return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private startProcessor(): void {
    setInterval(() => {
      this.processQueue();
    }, 1000); // Check queue every second
  }

  private async processQueue(): Promise<void> {
    // Process jobs if we have capacity and pending jobs
    while (this.currentlyProcessing < this.maxConcurrent && this.queue.length > 0) {
      const job = this.queue.shift();
      if (!job) continue;

      this.currentlyProcessing++;
      this.processing.set(job.id, job);
      this.updateStats();

      // Process job asynchronously
      this.processJob(job).catch((error) => {
        console.error(`[Queue] Unexpected error processing job ${job.id}:`, error);
        job.reject(error);
        this.completeJob(job.id, false);
      });
    }
  }

  private async processJob(job: QueueJob): Promise<void> {
    const startTime = Date.now();
    job.attempts++;

    try {
      if (process.env.NODE_ENV === "development") {
        console.log(
          `[Queue] Processing job ${job.id} (attempt ${job.attempts}/${job.maxAttempts})`,
        );
      }

      // Import the actual processing function (this would be your main analysis logic)
      const result = await this.executeAnalysis(job);

      const processingTime = Date.now() - startTime;
      this.recordProcessingTime(processingTime);

      job.resolve(result);
      this.completeJob(job.id, true);

      if (process.env.NODE_ENV === "development") {
        console.log(`[Queue] Completed job ${job.id} in ${processingTime}ms`);
      }
    } catch (error) {
      console.error(`[Queue] Job ${job.id} failed (attempt ${job.attempts}):`, error);

      if (job.attempts < job.maxAttempts) {
        // Retry with exponential backoff
        const delay = 2 ** job.attempts * 1000; // 2^attempts seconds
        setTimeout(() => {
          job.attempts--; // Reset for retry
          this.queue.unshift(job); // Add back to front of queue
          this.processing.delete(job.id);
          this.currentlyProcessing--;
          this.updateStats();
        }, delay);
      } else {
        // Max attempts reached, fail the job
        job.reject(error);
        this.completeJob(job.id, false);
      }
    }
  }

  private async executeAnalysis(job: QueueJob): Promise<{
    analysis: string;
    jobId: string;
    processedAt: string;
  }> {
    // This is a placeholder - in the actual implementation, this would call your analysis logic
    // For now, simulate processing time
    await new Promise((resolve) => setTimeout(resolve, Math.random() * 2000 + 1000));

    return {
      analysis: `Analysis for ${job.imageType} image`,
      jobId: job.id,
      processedAt: new Date().toISOString(),
    };
  }

  private completeJob(jobId: string, success: boolean): void {
    this.processing.delete(jobId);
    this.currentlyProcessing--;

    if (success) {
      this.stats.completed++;
    } else {
      this.stats.failed++;
    }

    this.stats.totalProcessed++;
    this.updateStats();
  }

  private recordProcessingTime(time: number): void {
    this.processingTimes.push(time);

    // Keep only last 100 processing times for average calculation
    if (this.processingTimes.length > 100) {
      this.processingTimes.shift();
    }

    this.stats.averageProcessingTime =
      this.processingTimes.reduce((sum, time) => sum + time, 0) / this.processingTimes.length;
  }

  private updateStats(): void {
    this.stats.pending = this.queue.length;
    this.stats.processing = this.processing.size;
  }

  // Get current queue statistics
  getStats(): QueueStats {
    return { ...this.stats };
  }

  // Get detailed queue information
  getQueueInfo() {
    return {
      stats: this.getStats(),
      queueSize: this.queue.length,
      processingJobs: Array.from(this.processing.values()).map((job) => ({
        id: job.id,
        imageType: job.imageType,
        priority: job.priority,
        attempts: job.attempts,
        age: Date.now() - job.createdAt.getTime(),
      })),
      pendingJobs: this.queue.map((job) => ({
        id: job.id,
        imageType: job.imageType,
        priority: job.priority,
        waitTime: Date.now() - job.createdAt.getTime(),
      })),
      configuration: {
        maxConcurrent: this.maxConcurrent,
        currentlyProcessing: this.currentlyProcessing,
      },
    };
  }

  // Estimate wait time for new jobs
  getEstimatedWaitTime(): number {
    const avgProcessingTime = this.stats.averageProcessingTime || 5000; // Default 5 seconds
    const queuePosition = this.queue.length;
    const slotsAvailable = Math.max(0, this.maxConcurrent - this.currentlyProcessing);

    if (slotsAvailable > queuePosition) {
      return 0; // Can process immediately
    }

    const jobsAhead = queuePosition - slotsAvailable;
    const estimatedWait = Math.ceil(jobsAhead / this.maxConcurrent) * avgProcessingTime;

    return estimatedWait;
  }

  // Update queue configuration
  updateConfig(maxConcurrent: number): void {
    if (maxConcurrent > 0 && maxConcurrent <= 10) {
      this.maxConcurrent = maxConcurrent;
      console.log(`[Queue] Updated max concurrent jobs to ${maxConcurrent}`);
    }
  }

  // Clear failed jobs
  clearFailedJobs(): void {
    this.stats.failed = 0;
    console.log("[Queue] Cleared failed job count");
  }

  // Get job status by ID
  getJobStatus(jobId: string): string {
    if (this.processing.has(jobId)) {
      return "processing";
    }

    if (this.queue.find((job) => job.id === jobId)) {
      return "pending";
    }

    return "unknown";
  }

  // Cancel a pending job
  cancelJob(jobId: string): boolean {
    const jobIndex = this.queue.findIndex((job) => job.id === jobId);
    if (jobIndex !== -1) {
      const job = this.queue[jobIndex];
      this.queue.splice(jobIndex, 1);
      job.reject(new Error("Job cancelled"));
      this.updateStats();
      return true;
    }
    return false;
  }
}

export const queueSystem = QueueSystem.getInstance();
export type { QueueJob, QueueStats };
