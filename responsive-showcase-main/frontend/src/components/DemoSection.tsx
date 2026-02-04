import {
  Camera,
  Square,
  Smile,
  Frown,
  Angry,
  AlertCircle,
  Zap,
  Meh,
  AlertTriangle,
  VideoOff,
  Upload,
  RotateCcw,
  Image as ImageIcon,
  Video,
  Play,
  Pause
} from "lucide-react";
import { useEffect, useCallback, useState, useRef } from "react";
import { Button } from "./ui/button";
import EmotionCard from "./EmotionCard";
import EmotionFeedback from "./EmotionFeedback";
import InputMethodSelector from "./InputMethodSelector";
import LoadingSpinner from "./LoadingSpinner";
import { useCamera } from "@/hooks/useCamera";
import * as faceapi from "face-api.js";

type InputMethod = "camera" | "image" | "video";

const emotions = [
  { name: "Happy", key: "Happy", icon: Smile, color: "bg-emotion-happy/20 text-emotion-happy" },
  { name: "Sad", key: "Sad", icon: Frown, color: "bg-emotion-sad/20 text-emotion-sad" },
  { name: "Angry", key: "Angry", icon: Angry, color: "bg-emotion-angry/20 text-emotion-angry" },
  { name: "Fear", key: "Fear", icon: AlertCircle, color: "bg-emotion-fear/20 text-emotion-fear" },
  { name: "Surprise", key: "Surprise", icon: Zap, color: "bg-emotion-surprise/20 text-emotion-surprise" },
  { name: "Neutral", key: "Neutral", icon: Meh, color: "bg-emotion-neutral/20 text-emotion-neutral" }
];

let faceLoaded = false;
async function loadFace() {
  if (!faceLoaded) {
    await faceapi.nets.tinyFaceDetector.loadFromUri("/models/face-api");
    faceLoaded = true;
  }
}

export default function DemoSection() {
  const [inputMethod, setInputMethod] = useState<InputMethod>("camera");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [result, setResult] = useState<any>(null);
  const [confidence, setConfidence] = useState(0);
  const [dominantEmotion, setDominantEmotion] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [videoPlaying, setVideoPlaying] = useState(false);

  const { videoRef, status, startCamera, stopCamera } = useCamera();
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const uploadedVideoRef = useRef<HTMLVideoElement>(null);

  // 🔥 BACKEND CALL
  const sendToBackend = async (blob: Blob) => {
  const formData = new FormData();
  formData.append("image", blob);

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    console.error("Backend error");
    return;
  }

  const data: {
    face_detected: boolean;
    emotion: string;
    confidence: number;
  } = await res.json();

  if (!data.face_detected) {
    setResult(null);
    setDominantEmotion(null);
    setConfidence(0);
    setShowFeedback(false);
    return;
  }

  // ✅ Build emotion map expected by UI
  const emotionMap = {
    Happy: 0,
    Sad: 0,
    Angry: 0,
    Fear: 0,
    Surprise: 0,
    Neutral: 0,
  };

  emotionMap[data.emotion as keyof typeof emotionMap] =
    Math.round(data.confidence);

  setResult(emotionMap);
  setDominantEmotion(data.emotion);
  setConfidence(Math.round(data.confidence));
  setShowFeedback(true);
};



  // 🖼 IMAGE
  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setProcessing(true);
    setPreviewUrl(URL.createObjectURL(file));

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      await loadFace();
      const det = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions());
      if (!det) {
        setProcessing(false);
        return;
      }

      const c = document.createElement("canvas");
      c.width = 48;
      c.height = 48;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, det.box.x, det.box.y, det.box.width, det.box.height, 0, 0, 48, 48);

      c.toBlob(async blob => {
        if (blob) await sendToBackend(blob);
        setProcessing(false);
      }, "image/jpeg");
    };
  };

  // 📷 CAMERA
  useEffect(() => {
    if (inputMethod !== "camera" || status !== "active" || !videoRef.current) return;

    const interval = setInterval(async () => {
      await loadFace();
      const det = await faceapi.detectSingleFace(videoRef.current!, new faceapi.TinyFaceDetectorOptions());
      if (!det) return;

      const c = document.createElement("canvas");
      c.width = 48;
      c.height = 48;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(videoRef.current!, det.box.x, det.box.y, det.box.width, det.box.height, 0, 0, 48, 48);

      c.toBlob(async blob => {
        if (blob) await sendToBackend(blob);
      }, "image/jpeg");
    }, 1500);

    return () => clearInterval(interval);
  }, [inputMethod, status]);

  // 🎥 VIDEO
  const handleVideoPlay = async () => {
    if (!uploadedVideoRef.current) return;
    setVideoPlaying(true);

    const interval = setInterval(async () => {
      await loadFace();
      const det = await faceapi.detectSingleFace(uploadedVideoRef.current!, new faceapi.TinyFaceDetectorOptions());
      if (!det) return;

      const c = document.createElement("canvas");
      c.width = 48;
      c.height = 48;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(uploadedVideoRef.current!, det.box.x, det.box.y, det.box.width, det.box.height, 0, 0, 48, 48);

      c.toBlob(async blob => {
        if (blob) await sendToBackend(blob);
      }, "image/jpeg");
    }, 2000);

    uploadedVideoRef.current.onpause = () => clearInterval(interval);
  };

  const reset = () => {
    stopCamera();
    setPreviewUrl(null);
    setResult(null);
    setDominantEmotion(null);
    setConfidence(0);
    setShowFeedback(false);
  };

  return (
    <section id="demo" className="py-24">
      <div className="container mx-auto px-4">
        <InputMethodSelector selectedMethod={inputMethod} onMethodChange={m => { reset(); setInputMethod(m); }} />

        <div className="relative aspect-video bg-secondary rounded-xl overflow-hidden mt-6">
          {inputMethod === "camera" && <video ref={videoRef} autoPlay muted className="w-full h-full object-cover scale-x-[-1]" />}
          {inputMethod === "image" && previewUrl && <img src={previewUrl} className="w-full h-full object-contain" />}
          {inputMethod === "video" && <video ref={uploadedVideoRef} className="w-full h-full" />}

          {processing && <LoadingSpinner size="lg" text="Analyzing..." />}
        </div>

        <div className="flex gap-4 mt-4">
          {inputMethod === "camera" && (
            <Button onClick={status === "active" ? stopCamera : startCamera}>
              {status === "active" ? <Square /> : <Camera />} {status === "active" ? "Stop" : "Start"}
            </Button>
          )}

          {inputMethod === "image" && (
            <Button onClick={() => imageInputRef.current?.click()}><Upload /> Upload</Button>
          )}

          {inputMethod === "video" && (
            <>
              <Button onClick={() => videoInputRef.current?.click()}><Upload /> Upload</Button>
              <Button onClick={handleVideoPlay}><Play /> Play</Button>
            </>
          )}
        </div>

        <input type="file" ref={imageInputRef} hidden accept="image/*" onChange={handleImageUpload} />
        <input type="file" ref={videoInputRef} hidden accept="video/*" onChange={e => {
          if (e.target.files?.[0]) uploadedVideoRef.current!.src = URL.createObjectURL(e.target.files[0]);
        }} />

        <div className="grid grid-cols-2 gap-4 mt-8">
          {emotions.map(e => (
            <EmotionCard
              key={e.key}
              name={e.name}
              icon={e.icon}
              percentage={result ? result[e.key] || 0 : 0}
              color={e.color}
              isActive={dominantEmotion === e.key} description={""}            />
          ))}
        </div>

        {showFeedback && dominantEmotion && (
          <EmotionFeedback emotion={dominantEmotion} confidence={confidence} onClose={() => setShowFeedback(false)} />
        )}
      </div>
    </section>
  );
}
