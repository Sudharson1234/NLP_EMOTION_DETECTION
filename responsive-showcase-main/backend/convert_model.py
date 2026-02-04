<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EmotiSense</title>

<style>
body {
  background: #050a14;
  color: white;
  font-family: Arial, sans-serif;
}
button { padding:10px; margin:4px; cursor:pointer; }
.bar { height:10px; background:#222; border-radius:6px; overflow:hidden; }
.bar div { height:100%; width:0%; background:#33d6ff; transition:0.3s; }
.emotion-row { margin-bottom:12px; }
video, img { width:100%; max-height:320px; }
</style>
</head>

<body>

<h2>EmotiSense – Emotion Detection</h2>

<div>
  <input type="file" id="fileInput" hidden accept="image/*">
  <button id="btnUpload">Upload Image</button>
  <button id="btnPredict" disabled>Predict</button>
</div>

<div>
  <button id="btnStartCam">Start Camera</button>
  <button id="btnStopCam" disabled>Stop Camera</button>
</div>

<div>
  <input type="file" id="videoInput" hidden accept="video/*">
  <button id="btnUploadVideo">Upload Video</button>
  <button id="btnDetectVideo" disabled>Detect Video</button>
  <button id="btnStopVideo" disabled>Stop Video</button>
</div>

<p id="statusText">Backend: http://127.0.0.1:5000</p>

<img id="previewImg" style="display:none">
<video id="cameraVideo" autoplay muted playsinline style="display:none"></video>
<video id="videoPreview" playsinline style="display:none"></video>

<hr>

<div id="emotionList"></div>

<script>
const BACKEND_URL = "http://127.0.0.1:5000/predict";

const emotions = ["Angry","Fear","Happy","Neutral","Sad","Surprise"];

const fileInput = document.getElementById("fileInput");
const videoInput = document.getElementById("videoInput");

const previewImg = document.getElementById("previewImg");
const cameraVideo = document.getElementById("cameraVideo");
const videoPreview = document.getElementById("videoPreview");

const btnUpload = document.getElementById("btnUpload");
const btnPredict = document.getElementById("btnPredict");

const btnStartCam = document.getElementById("btnStartCam");
const btnStopCam = document.getElementById("btnStopCam");

const btnUploadVideo = document.getElementById("btnUploadVideo");
const btnDetectVideo = document.getElementById("btnDetectVideo");
const btnStopVideo = document.getElementById("btnStopVideo");

const statusText = document.getElementById("statusText");
const emotionList = document.getElementById("emotionList");

let currentFile = null;
let camStream = null;
let camInterval = null;
let vidInterval = null;

/* ---------------- UI ---------------- */

function createEmotionBars(){
  emotionList.innerHTML="";
  emotions.forEach(e=>{
    const row=document.createElement("div");
    row.className="emotion-row";
    row.innerHTML=`
      <b>${e}: <span id="val-${e}">0%</span></b>
      <div class="bar"><div id="bar-${e}"></div></div>
    `;
    emotionList.appendChild(row);
  });
}
createEmotionBars();

function resetBars(){
  emotions.forEach(e=>{
    document.getElementById(`val-${e}`).textContent="0%";
    document.getElementById(`bar-${e}`).style.width="0%";
  });
}

function hideAll(){
  previewImg.style.display="none";
  cameraVideo.style.display="none";
  videoPreview.style.display="none";
}

/* ---------------- HELPERS ---------------- */

async function sendBlob(blob){
  const fd=new FormData();
  fd.append("image",blob,"frame.jpg");
  const res=await fetch(BACKEND_URL,{method:"POST",body:fd});
  return await res.json();
}

async function saveFrame(blob,emotion,confidence){
  const fd=new FormData();
  fd.append("image",blob,"frame.jpg");
  fd.append("emotion",emotion);
  fd.append("confidence",confidence);
  await fetch("http://127.0.0.1:5000/capture",{method:"POST",body:fd});
}

async function captureFrame(video){
  const c=document.createElement("canvas");
  c.width=640; c.height=640;
  c.getContext("2d").drawImage(video,0,0,640,640);
  return await new Promise(r=>c.toBlob(r,"image/jpeg",0.9));
}

/* ---------------- IMAGE ---------------- */

btnUpload.onclick=()=>fileInput.click();

fileInput.onchange=()=>{
  const f=fileInput.files[0];
  if(!f)return;
  currentFile=f;
  hideAll();
  previewImg.src=URL.createObjectURL(f);
  previewImg.style.display="block";
  btnPredict.disabled=false;
  resetBars();
};

btnPredict.onclick=async()=>{
  const fd=new FormData();
  fd.append("image",currentFile);
  const res=await fetch(BACKEND_URL,{method:"POST",body:fd});
  const data=await res.json();
  resetBars();
  if(!data.face_detected)return;
  const e=data.emotion;
  const c=Math.round(data.confidence);
  document.getElementById(`val-${e}`).textContent=c+"%";
  document.getElementById(`bar-${e}`).style.width=c+"%";
};

/* ---------------- CAMERA ---------------- */

btnStartCam.onclick=async()=>{
  camStream=await navigator.mediaDevices.getUserMedia({video:true});
  hideAll();
  cameraVideo.srcObject=camStream;
  cameraVideo.style.display="block";
  btnStopCam.disabled=false;
  btnStartCam.disabled=true;

  camInterval=setInterval(async()=>{
    const blob=await captureFrame(cameraVideo);
    const data=await sendBlob(blob);
    if(!data.face_detected)return;
    const e=data.emotion;
    const c=Math.round(data.confidence);
    resetBars();
    await saveFrame(blob,e,c);
    document.getElementById(`val-${e}`).textContent=c+"%";
    document.getElementById(`bar-${e}`).style.width=c+"%";
  },3000);
};

btnStopCam.onclick=()=>{
  camStream.getTracks().forEach(t=>t.stop());
  clearInterval(camInterval);
  hideAll();
  btnStartCam.disabled=false;
  btnStopCam.disabled=true;
};

/* ---------------- VIDEO ---------------- */

btnUploadVideo.onclick=()=>videoInput.click();

videoInput.onchange=()=>{
  hideAll();
  videoPreview.src=URL.createObjectURL(videoInput.files[0]);
  videoPreview.style.display="block";
  btnDetectVideo.disabled=false;
};

btnDetectVideo.onclick=async()=>{
  await videoPreview.play();
  btnStopVideo.disabled=false;
  btnDetectVideo.disabled=true;

  vidInterval=setInterval(async()=>{
    if(videoPreview.paused||videoPreview.ended)return;
    const blob=await captureFrame(videoPreview);
    const data=await sendBlob(blob);
    if(!data.face_detected)return;
    const e=data.emotion;
    const c=Math.round(data.confidence);
    resetBars();
    await saveFrame(blob,e,c);
    document.getElementById(`val-${e}`).textContent=c+"%";
    document.getElementById(`bar-${e}`).style.width=c+"%";
  },4000);
};

btnStopVideo.onclick=()=>{
  clearInterval(vidInterval);
  videoPreview.pause();
  btnStopVideo.disabled=true;
  btnDetectVideo.disabled=false;
};
</script>

</body>
</html>
