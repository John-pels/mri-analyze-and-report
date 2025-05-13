"use client";

import { useDropzone } from "react-dropzone";

interface UseDropzoneHandlerProps {
  onImageUpload: (file: File) => void;
  maxFiles?: number;
}

export function useDropzoneHandler({
  onImageUpload,
  maxFiles = 1,
}: UseDropzoneHandlerProps) {
  const onDrop = (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    onImageUpload(file);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [],
    },
    maxFiles,
  });

  return {
    getRootProps,
    getInputProps,
    isDragActive,
  };
}
