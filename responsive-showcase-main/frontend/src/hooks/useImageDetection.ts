import { useState } from "react";
import { predictEmotionFromImage, EmotionKey } from "@/lib/emotionPredict";

interface DetectionResult {
  faceDetected: boolean;
  dominantEmotion?: EmotionKey;
  confidence?: number;
  emotions?: Record<EmotionKey, number>;
}

export function useImageDetection() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);

  async function processImage(file: File) {
    setIsProcessing(true);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.src = url;
    await new Promise((res) => (img.onload = res));

    const res = await predictEmotionFromImage(img);
    setResult(res);
    setIsProcessing(false);
  }

  function clearResult() {
    setPreviewUrl(null);
    setResult(null);
  }

  return {
    isProcessing,
    previewUrl,
    result,
    processImage,
    clearResult,
  };
}
