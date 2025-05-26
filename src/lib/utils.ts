import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${Number.parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`;
}

export function getColorByConfidence(confidence: number): string {
  if (confidence >= 0.9) return "#10b981"; // Green for high confidence
  if (confidence >= 0.7) return "#f59e0b"; // Yellow for medium confidence
  return "#ef4444"; // Red for low confidence
}

export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number,
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout !== null) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

export function validateImageFile(file: File): {
  valid: boolean;
  message?: string;
} {
  // Check if it's an image
  if (!file.type.startsWith("image/")) {
    return { valid: false, message: "File must be an image" };
  }

  // Check file size (3MB limit)
  const MAX_FILE_SIZE = 3 * 1024 * 1024; // 3MB in bytes
  if (file.size > MAX_FILE_SIZE) {
    return { valid: false, message: "Image must be smaller than 3MB" };
  }

  return { valid: true };
}
