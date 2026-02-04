import { Camera, Square, Smile, Frown, Angry, AlertCircle, Zap, Meh, AlertTriangle, VideoOff, Upload, Play, Pause, RotateCcw, Image as ImageIcon, Video } from 'lucide-react';
import { useEffect, useCallback, useState, useRef } from 'react';
import { Button } from './ui/button';
import EmotionCard from './EmotionCard';
import EmotionFeedback from './EmotionFeedback';
import InputMethodSelector from './InputMethodSelector';
import LoadingSpinner from './LoadingSpinner';
import { useCamera } from '@/hooks/useCamera';
import { useEmotionDetection } from '@/hooks/useEmotionDetection';
import { useImageDetection } from '@/hooks/useImageDetection';
import { useVideoDetection } from '@/hooks/useVideoDetection';

type InputMethod = 'camera' | 'image' | 'video';

const emotions = [
  { 
    name: 'Happy', 
    key: 'happy',
    icon: Smile, 
    color: 'bg-emotion-happy/20 text-emotion-happy',
    barColor: 'bg-emotion-happy',
    description: 'Positive emotional state showing joy and contentment' 
  },
  { 
    name: 'Sad', 
    key: 'sad',
    icon: Frown, 
    color: 'bg-emotion-sad/20 text-emotion-sad',
    barColor: 'bg-emotion-sad',
    description: 'Feelings of sorrow, unhappiness, or disappointment' 
  },
  { 
    name: 'Angry', 
    key: 'angry',
    icon: Angry, 
    color: 'bg-emotion-angry/20 text-emotion-angry',
    barColor: 'bg-emotion-angry',
    description: 'Strong feeling of displeasure or hostility' 
  },
  { 
    name: 'Fear', 
    key: 'fearful',
    icon: AlertCircle, 
    color: 'bg-emotion-fear/20 text-emotion-fear',
    barColor: 'bg-emotion-fear',
    description: 'Response to perceived threat or danger' 
  },
  { 
    name: 'Surprise', 
    key: 'surprised',
    icon: Zap, 
    color: 'bg-emotion-surprise/20 text-emotion-surprise',
    barColor: 'bg-emotion-surprise',
    description: 'Brief emotional state from unexpected events' 
  },
  { 
    name: 'Neutral', 
    key: 'neutral',
    icon: Meh, 
    color: 'bg-emotion-neutral/20 text-emotion-neutral',
    barColor: 'bg-emotion-neutral',
    description: 'Baseline state without strong emotional expression' 
  },
];

const DemoSection = () => {
  const [inputMethod, setInputMethod] = useState<InputMethod>('camera');
  const [showFeedback, setShowFeedback] = useState(false);
  
  // Camera hook
  const { videoRef, status: cameraStatus, error: cameraError, startCamera, stopCamera } = useCamera();
  const { 
    isModelLoading: isCameraModelLoading, 
    isModelLoaded: isCameraModelLoaded, 
    modelLoadError: cameraModelError, 
    detection: cameraDetection, 
    startDetection, 
    stopDetection 
  } = useEmotionDetection();

  // Image hook
  const {
    isModelLoading: isImageModelLoading,
    isProcessing: isImageProcessing,
    result: imageResult,
    previewUrl: imagePreviewUrl,
    processImage,
    clearResult: clearImageResult
  } = useImageDetection();

  // Video hook
  const {
    isModelLoading: isVideoModelLoading,
    isModelLoaded: isVideoModelLoaded,
    isProcessing: isVideoProcessing,
    isPlaying: isVideoPlaying,
    result: videoResult,
    videoUrl,
    processVideo,
    startVideoDetection,
    stopVideoDetection,
    clearVideo
  } = useVideoDetection();

  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const uploadedVideoRef = useRef<HTMLVideoElement>(null);

  // Determine current state based on input method
  const isActive = inputMethod === 'camera' ? cameraStatus === 'active' : 
                   inputMethod === 'image' ? !!imagePreviewUrl :
                   inputMethod === 'video' ? !!videoUrl : false;
  
  const isLoading = inputMethod === 'camera' ? (cameraStatus === 'requesting' || isCameraModelLoading) :
                    inputMethod === 'image' ? (isImageModelLoading || isImageProcessing) :
                    inputMethod === 'video' ? (isVideoModelLoading || isVideoProcessing) : false;

  const currentDetection = inputMethod === 'camera' ? cameraDetection :
                           inputMethod === 'image' ? imageResult :
                           inputMethod === 'video' ? videoResult : null;

  const handleStartCamera = useCallback(async () => {
    await startCamera();
  }, [startCamera]);

  const handleStopCamera = useCallback(() => {
    stopCamera();
    stopDetection();
    setShowFeedback(false);
  }, [stopCamera, stopDetection]);

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processImage(file);
    }
  }, [processImage]);

  const handleVideoUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      processVideo(file);
    }
  }, [processVideo]);

  const handleVideoPlay = useCallback(() => {
    if (uploadedVideoRef.current) {
      uploadedVideoRef.current.play();
      startVideoDetection(uploadedVideoRef.current);
    }
  }, [startVideoDetection]);

  const handleVideoPause = useCallback(() => {
    if (uploadedVideoRef.current) {
      uploadedVideoRef.current.pause();
      stopVideoDetection();
    }
  }, [stopVideoDetection]);

  const handleReset = useCallback(() => {
    if (inputMethod === 'camera') {
      handleStopCamera();
    } else if (inputMethod === 'image') {
      clearImageResult();
    } else if (inputMethod === 'video') {
      clearVideo();
    }
    setShowFeedback(false);
  }, [inputMethod, handleStopCamera, clearImageResult, clearVideo]);

  const handleMethodChange = useCallback((method: InputMethod) => {
    handleReset();
    setInputMethod(method);
  }, [handleReset]);

  // Start detection when camera is active
  useEffect(() => {
    if (inputMethod === 'camera' && cameraStatus === 'active' && videoRef.current) {
      startDetection(videoRef.current);
    }
  }, [inputMethod, cameraStatus, videoRef, startDetection]);

  // Show feedback when emotion is detected
  useEffect(() => {
    if (currentDetection?.faceDetected && currentDetection.confidence > 50) {
      setShowFeedback(true);
    }
  }, [currentDetection]);

  // Get emotion data for cards
  const getEmotionPercentage = (key: string): number => {
    if (!currentDetection || !currentDetection.faceDetected) return 0;
    return currentDetection.emotions[key as keyof typeof currentDetection.emotions] || 0;
  };

  const getDominantEmotionIndex = (): number => {
    if (!currentDetection || !currentDetection.faceDetected) return -1;
    return emotions.findIndex(e => e.key === currentDetection.dominantEmotion);
  };

  const dominantIndex = getDominantEmotionIndex();
  const dominantEmotion = dominantIndex >= 0 ? emotions[dominantIndex] : null;

  return (
    <section id="demo" className="py-24 relative">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="font-display text-3xl md:text-5xl font-bold mb-4">
            Live <span className="text-gradient">Emotion Detection</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
            Experience real-time facial emotion recognition. Use your camera, 
            upload an image, or analyze a video file.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Detection View */}
          <div className="glass-card rounded-3xl p-6 border border-border/50">
            {/* Input Method Selector */}
            <div className="mb-6">
              <InputMethodSelector 
                selectedMethod={inputMethod} 
                onMethodChange={handleMethodChange}
                disabled={isLoading}
              />
            </div>

            <div className="relative aspect-video bg-secondary/50 rounded-2xl overflow-hidden">
              {/* Camera Video Element */}
              {inputMethod === 'camera' && (
                <video
                  ref={videoRef}
                  className={`absolute inset-0 w-full h-full object-cover scale-x-[-1] ${cameraStatus === 'active' ? 'block' : 'hidden'}`}
                  playsInline
                  muted
                />
              )}

              {/* Image Preview */}
              {inputMethod === 'image' && imagePreviewUrl && (
                <img
                  src={imagePreviewUrl}
                  alt="Uploaded for analysis"
                  className="absolute inset-0 w-full h-full object-contain"
                />
              )}

              {/* Video Preview */}
              {inputMethod === 'video' && videoUrl && (
                <video
                  ref={uploadedVideoRef}
                  src={videoUrl}
                  className="absolute inset-0 w-full h-full object-contain"
                  playsInline
                  onEnded={stopVideoDetection}
                />
              )}

              {/* Idle States */}
              {inputMethod === 'camera' && cameraStatus === 'idle' && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Click "Start Camera" to begin detection</p>
                  </div>
                </div>
              )}

              {inputMethod === 'image' && !imagePreviewUrl && !isLoading && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <ImageIcon className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Click "Upload Image" to analyze</p>
                  </div>
                </div>
              )}

              {inputMethod === 'video' && !videoUrl && !isLoading && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <Video className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Click "Upload Video" to analyze</p>
                  </div>
                </div>
              )}

              {/* Loading State */}
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
                  <LoadingSpinner 
                    size="lg" 
                    text={
                      inputMethod === 'camera' && cameraStatus === 'requesting' ? 'Requesting camera access...' :
                      'Loading AI models...'
                    } 
                  />
                </div>
              )}

              {/* Error State - Camera */}
              {inputMethod === 'camera' && (cameraStatus === 'error' || cameraStatus === 'denied') && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center px-4">
                    <AlertTriangle className="w-16 h-16 text-destructive mx-auto mb-4" />
                    <p className="text-destructive font-medium mb-2">Camera Access Error</p>
                    <p className="text-muted-foreground text-sm max-w-xs">{cameraError}</p>
                  </div>
                </div>
              )}

              {/* Model Load Error */}
              {cameraModelError && inputMethod === 'camera' && cameraStatus === 'active' && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
                  <div className="text-center px-4">
                    <AlertTriangle className="w-16 h-16 text-destructive mx-auto mb-4" />
                    <p className="text-destructive font-medium mb-2">Model Loading Error</p>
                    <p className="text-muted-foreground text-sm max-w-xs">{cameraModelError}</p>
                  </div>
                </div>
              )}

              {/* Active State Overlays */}
              {isActive && !isLoading && (
                <>
                  {/* Loading models overlay for camera */}
                  {inputMethod === 'camera' && isCameraModelLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm">
                      <LoadingSpinner size="lg" text="Loading AI models..." />
                    </div>
                  )}

                  {/* No face detected warning */}
                  {currentDetection && !currentDetection.faceDetected && (
                    <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-destructive/80 text-destructive-foreground rounded-full text-sm font-medium flex items-center gap-2">
                      <VideoOff className="w-4 h-4" />
                      No face detected
                    </div>
                  )}

                  {/* Face Detection Box */}
                  {currentDetection?.faceDetected && (
                    <>
                      <div className="absolute inset-12 border-2 border-primary rounded-lg animate-pulse-glow pointer-events-none">
                        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-primary text-primary-foreground text-xs font-semibold rounded-full">
                          Face Detected
                        </div>
                      </div>

                      {/* Scan Line */}
                      {inputMethod === 'camera' && (
                        <div className="absolute inset-0 overflow-hidden pointer-events-none">
                          <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent animate-scan-line" />
                        </div>
                      )}
                    </>
                  )}

                  {/* Dominant Emotion Badge */}
                  {currentDetection?.faceDetected && dominantEmotion && (
                    <div className="absolute bottom-4 left-4 right-4 glass-card rounded-xl p-4 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <dominantEmotion.icon className={`w-8 h-8 ${dominantEmotion.color.split(' ')[1]}`} />
                        <div>
                          <p className="text-xs text-muted-foreground">Detected Emotion</p>
                          <p className="font-display font-bold text-lg">{dominantEmotion.name}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">Confidence</p>
                        <p className="font-display font-bold text-lg text-primary">{currentDetection.confidence}%</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Hidden file inputs */}
            <input
              type="file"
              ref={imageInputRef}
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <input
              type="file"
              ref={videoInputRef}
              accept="video/*"
              onChange={handleVideoUpload}
              className="hidden"
            />

            {/* Controls */}
            <div className="flex gap-4 mt-6">
              {/* Camera Controls */}
              {inputMethod === 'camera' && (
                <>
                  {cameraStatus !== 'active' ? (
                    <Button 
                      className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow-box font-semibold"
                      onClick={handleStartCamera}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          Connecting...
                        </>
                      ) : (
                        <>
                          <Camera className="w-5 h-5 mr-2" />
                          Start Camera
                        </>
                      )}
                    </Button>
                  ) : (
                    <Button 
                      className="flex-1 bg-destructive text-destructive-foreground hover:bg-destructive/90 font-semibold"
                      onClick={handleStopCamera}
                    >
                      <Square className="w-5 h-5 mr-2" />
                      Stop Detection
                    </Button>
                  )}
                </>
              )}

              {/* Image Controls */}
              {inputMethod === 'image' && (
                <>
                  {!imagePreviewUrl ? (
                    <Button 
                      className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow-box font-semibold"
                      onClick={() => imageInputRef.current?.click()}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          Processing...
                        </>
                      ) : (
                        <>
                          <Upload className="w-5 h-5 mr-2" />
                          Upload Image
                        </>
                      )}
                    </Button>
                  ) : (
                    <>
                      <Button 
                        className="flex-1"
                        variant="outline"
                        onClick={() => imageInputRef.current?.click()}
                        disabled={isLoading}
                      >
                        <Upload className="w-5 h-5 mr-2" />
                        Upload New
                      </Button>
                      <Button 
                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        onClick={clearImageResult}
                      >
                        <RotateCcw className="w-5 h-5" />
                      </Button>
                    </>
                  )}
                </>
              )}

              {/* Video Controls */}
              {inputMethod === 'video' && (
                <>
                  {!videoUrl ? (
                    <Button 
                      className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow-box font-semibold"
                      onClick={() => videoInputRef.current?.click()}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <LoadingSpinner size="sm" className="mr-2" />
                          Loading...
                        </>
                      ) : (
                        <>
                          <Upload className="w-5 h-5 mr-2" />
                          Upload Video
                        </>
                      )}
                    </Button>
                  ) : (
                    <>
                      {!isVideoPlaying ? (
                        <Button 
                          className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow-box font-semibold"
                          onClick={handleVideoPlay}
                          disabled={isLoading}
                        >
                          <Play className="w-5 h-5 mr-2" />
                          Play & Detect
                        </Button>
                      ) : (
                        <Button 
                          className="flex-1 bg-secondary text-secondary-foreground hover:bg-secondary/90 font-semibold"
                          onClick={handleVideoPause}
                        >
                          <Pause className="w-5 h-5 mr-2" />
                          Pause
                        </Button>
                      )}
                      <Button 
                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        onClick={clearVideo}
                      >
                        <RotateCcw className="w-5 h-5" />
                      </Button>
                    </>
                  )}
                </>
              )}
            </div>

            {/* Status indicator */}
            <div className="mt-4 flex items-center justify-center gap-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${isActive && currentDetection?.faceDetected ? 'bg-green-500 animate-pulse' : 'bg-muted'}`} />
              <span className="text-muted-foreground">
                {inputMethod === 'camera' && !isActive && 'Camera inactive'}
                {inputMethod === 'camera' && isActive && isCameraModelLoading && 'Loading AI models...'}
                {inputMethod === 'camera' && isActive && isCameraModelLoaded && currentDetection?.faceDetected && 'Detecting emotions in real-time'}
                {inputMethod === 'camera' && isActive && isCameraModelLoaded && !currentDetection?.faceDetected && 'Waiting for face...'}
                {inputMethod === 'image' && !imagePreviewUrl && 'Ready to analyze image'}
                {inputMethod === 'image' && imagePreviewUrl && imageResult?.faceDetected && 'Emotion detected'}
                {inputMethod === 'image' && imagePreviewUrl && !imageResult?.faceDetected && 'No face found in image'}
                {inputMethod === 'video' && !videoUrl && 'Ready to analyze video'}
                {inputMethod === 'video' && videoUrl && isVideoPlaying && 'Analyzing video...'}
                {inputMethod === 'video' && videoUrl && !isVideoPlaying && 'Video ready - Press play to analyze'}
              </span>
            </div>
          </div>

          {/* Right Column - Emotion Cards + Feedback */}
          <div className="space-y-6">
            {/* Emotion Cards Grid */}
            <div className="grid grid-cols-2 gap-4">
              {emotions.map((emotion, index) => (
                <EmotionCard
                  key={emotion.name}
                  name={emotion.name}
                  icon={emotion.icon}
                  percentage={getEmotionPercentage(emotion.key)}
                  color={emotion.color}
                  description={emotion.description}
                  isActive={isActive && currentDetection?.faceDetected && index === dominantIndex}
                  delay={index * 100}
                />
              ))}
            </div>

            {/* Emotion Feedback Chat */}
            {showFeedback && currentDetection?.faceDetected && (
              <EmotionFeedback
                emotion={currentDetection.dominantEmotion}
                confidence={currentDetection.confidence}
                onClose={() => setShowFeedback(false)}
              />
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default DemoSection;
