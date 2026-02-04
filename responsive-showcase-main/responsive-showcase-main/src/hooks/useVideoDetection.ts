import { useState, useCallback, useRef } from 'react';
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

export interface VideoDetectionResult {
  emotions: EmotionData;
  dominantEmotion: string;
  confidence: number;
  faceDetected: boolean;
}

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

export const useVideoDetection = () => {
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [result, setResult] = useState<VideoDetectionResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectionIntervalRef = useRef<number | null>(null);

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

  const detectEmotions = useCallback(async () => {
    if (!videoRef.current || !isModelLoaded) return;
    
    const video = videoRef.current;
    
    if (video.paused || video.ended || video.readyState < 2) return;
    
    try {
      const detections = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
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
        
        setResult({
          emotions,
          dominantEmotion,
          confidence,
          faceDetected: true
        });
      } else {
        setResult(prev => prev ? { ...prev, faceDetected: false } : null);
      }
    } catch (err) {
      console.error('Detection error:', err);
    }
  }, [isModelLoaded]);

  const processVideo = useCallback(async (file: File): Promise<void> => {
    setIsProcessing(true);
    setResult(null);
    
    // Create video URL
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    // Load models
    await loadModels();
    setIsProcessing(false);
  }, [loadModels]);

  const startVideoDetection = useCallback((video: HTMLVideoElement) => {
    videoRef.current = video;
    setIsPlaying(true);
    
    // Clear any existing interval
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
    
    // Start detection loop
    detectionIntervalRef.current = window.setInterval(() => {
      detectEmotions();
    }, 200);
  }, [detectEmotions]);

  const stopVideoDetection = useCallback(() => {
    setIsPlaying(false);
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
  }, []);

  const clearVideo = useCallback(() => {
    stopVideoDetection();
    setResult(null);
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);
    }
  }, [videoUrl, stopVideoDetection]);

  return {
    isModelLoading,
    isModelLoaded,
    modelLoadError,
    isProcessing,
    isPlaying,
    result,
    videoUrl,
    processVideo,
    startVideoDetection,
    stopVideoDetection,
    clearVideo
  };
};
