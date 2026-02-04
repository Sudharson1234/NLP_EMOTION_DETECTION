import { useState, useEffect, useCallback, useRef } from 'react';
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

export interface DetectionResult {
  emotions: EmotionData;
  dominantEmotion: string;
  confidence: number;
  faceDetected: boolean;
}

interface UseEmotionDetectionReturn {
  isModelLoading: boolean;
  isModelLoaded: boolean;
  modelLoadError: string | null;
  detection: DetectionResult | null;
  isDetecting: boolean;
  startDetection: (video: HTMLVideoElement) => void;
  stopDetection: () => void;
}

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

export const useEmotionDetection = (): UseEmotionDetectionReturn => {
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  
  const detectionIntervalRef = useRef<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Load face-api models
  const loadModels = useCallback(async () => {
    if (isModelLoaded || isModelLoading) return;
    
    try {
      setIsModelLoading(true);
      setModelLoadError(null);
      
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
      ]);
      
      setIsModelLoaded(true);
    } catch (err) {
      console.error('Failed to load face-api models:', err);
      setModelLoadError('Failed to load emotion detection models. Please check your internet connection.');
    } finally {
      setIsModelLoading(false);
    }
  }, [isModelLoaded, isModelLoading]);

  // Detect emotions from video
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
        
        // Map to our emotion structure (percentage)
        const emotions: EmotionData = {
          neutral: Math.round(expressions.neutral * 100),
          happy: Math.round(expressions.happy * 100),
          sad: Math.round(expressions.sad * 100),
          angry: Math.round(expressions.angry * 100),
          fearful: Math.round(expressions.fearful * 100),
          disgusted: Math.round(expressions.disgusted * 100),
          surprised: Math.round(expressions.surprised * 100)
        };
        
        // Find dominant emotion
        const emotionEntries = Object.entries(emotions) as [string, number][];
        const [dominantEmotion, confidence] = emotionEntries.reduce(
          (max, curr) => curr[1] > max[1] ? curr : max,
          ['neutral', 0]
        );
        
        setDetection({
          emotions,
          dominantEmotion,
          confidence,
          faceDetected: true
        });
      } else {
        setDetection(prev => prev ? { ...prev, faceDetected: false } : null);
      }
    } catch (err) {
      console.error('Detection error:', err);
    }
  }, [isModelLoaded]);

  const startDetection = useCallback((video: HTMLVideoElement) => {
    videoRef.current = video;
    setIsDetecting(true);
    
    // Load models if not loaded
    loadModels();
    
    // Clear any existing interval
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
    
    // Start detection loop
    detectionIntervalRef.current = window.setInterval(() => {
      detectEmotions();
    }, 200); // Detect every 200ms for smooth updates
  }, [loadModels, detectEmotions]);

  const stopDetection = useCallback(() => {
    setIsDetecting(false);
    setDetection(null);
    videoRef.current = null;
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, []);

  return {
    isModelLoading,
    isModelLoaded,
    modelLoadError,
    detection,
    isDetecting,
    startDetection,
    stopDetection
  };
};
