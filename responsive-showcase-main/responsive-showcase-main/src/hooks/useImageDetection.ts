import { useState, useCallback } from 'react';
import * as faceapi from 'face-api.js';

export interface EmotionData {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
}

export interface ImageDetectionResult {
  emotions: EmotionData;
  dominantEmotion: string;
  confidence: number;
  faceDetected: boolean;
}

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

export const useImageDetection = () => {
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ImageDetectionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    if (isModelLoaded || isModelLoading) return true;
    
    try {
      setIsModelLoading(true);
      setModelLoadError(null);
      
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
      ]);
      
      setIsModelLoaded(true);
      return true;
    } catch (err) {
      console.error('Failed to load face-api models:', err);
      setModelLoadError('Failed to load emotion detection models.');
      return false;
    } finally {
      setIsModelLoading(false);
    }
  }, [isModelLoaded, isModelLoading]);

  const processImage = useCallback(async (file: File): Promise<ImageDetectionResult | null> => {
    setIsProcessing(true);
    setResult(null);
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    try {
      // Load models if not loaded
      const modelsLoaded = await loadModels();
      if (!modelsLoaded) {
        setIsProcessing(false);
        return null;
      }

      // Create image element
      const img = new Image();
      img.src = url;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      // Detect emotions
      const detections = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }))
        .withFaceExpressions();

      if (detections) {
        const expressions = detections.expressions;
        
        const emotions: EmotionData = {
          neutral: Math.round(expressions.neutral * 100),
          happy: Math.round(expressions.happy * 100),
          sad: Math.round(expressions.sad * 100),
          angry: Math.round(expressions.angry * 100),
          fearful: Math.round(expressions.fearful * 100),
          disgusted: Math.round(expressions.disgusted * 100),
          surprised: Math.round(expressions.surprised * 100)
        };
        
        const emotionEntries = Object.entries(emotions) as [string, number][];
        const [dominantEmotion, confidence] = emotionEntries.reduce(
          (max, curr) => curr[1] > max[1] ? curr : max,
          ['neutral', 0]
        );
        
        const detectionResult = {
          emotions,
          dominantEmotion,
          confidence,
          faceDetected: true
        };
        
        setResult(detectionResult);
        setIsProcessing(false);
        return detectionResult;
      } else {
        const noFaceResult = {
          emotions: { neutral: 0, happy: 0, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0 },
          dominantEmotion: 'neutral',
          confidence: 0,
          faceDetected: false
        };
        setResult(noFaceResult);
        setIsProcessing(false);
        return noFaceResult;
      }
    } catch (err) {
      console.error('Image processing error:', err);
      setIsProcessing(false);
      return null;
    }
  }, [loadModels]);

  const clearResult = useCallback(() => {
    setResult(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  return {
    isModelLoading,
    isModelLoaded,
    modelLoadError,
    isProcessing,
    result,
    previewUrl,
    processImage,
    clearResult
  };
};
