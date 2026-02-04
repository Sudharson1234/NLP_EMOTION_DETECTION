import * as faceapi from "face-api.js";

let faceModelsLoaded = false;

async function loadFaceModel() {
  if (!faceModelsLoaded) {
    await faceapi.nets.tinyFaceDetector.loadFromUri("/models/face-api");
    faceModelsLoaded = true;
  }
}

export type EmotionKey =
  | "happy"
  | "sad"
  | "angry"
  | "fear"
  | "surprise"
  | "neutral";

export async function predictEmotionFromImage(img: HTMLImageElement) {
  await loadFaceModel();

  const detection = await faceapi.detectSingleFace(
    img,
    new faceapi.TinyFaceDetectorOptions()
  );

  if (!detection) {
    return { faceDetected: false };
  }

  const { x, y, width, height } = detection.box;

  const canvas = document.createElement("canvas");
  canvas.width = 48;
  canvas.height = 48;

  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, x, y, width, height, 0, 0, 48, 48);

  const blob: Blob = await new Promise((resolve) =>
    canvas.toBlob((b) => resolve(b!), "image/jpeg")
  );

  const formData = new FormData();
  formData.append("image", blob, "face.jpg");

  const response = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();

  if (!result.face_detected) {
    return { faceDetected: false };
  }

  const dominantEmotion = result.emotion.toLowerCase() as EmotionKey;

  const emotions: Record<EmotionKey, number> = {
    happy: 0,
    sad: 0,
    angry: 0,
    fear: 0,
    surprise: 0,
    neutral: 0,
  };

  emotions[dominantEmotion] = Math.round(result.confidence);

  return {
    faceDetected: true,
    dominantEmotion,
    confidence: Math.round(result.confidence),
    emotions,
  };
}
