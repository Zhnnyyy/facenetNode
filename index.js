require("@tensorflow/tfjs-core/dist/ops/softmax.js");
const express = require("express");
const faceapi = require("face-api.js");
const tf = require("@tensorflow/tfjs-node"); // Use tfjs-node for better performance
const canvas = require("canvas");
const path = require("path");
const fs = require("fs");

// Initialize canvas for face-api.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const port = 3000;

app.use(express.json({ limit: "10mb" }));

// Load models
const MODEL_URL = path.join(__dirname, "models");

async function loadModels() {
  await faceapi.nets.mtcnn.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
}

// Convert base64 to buffer
function base64ToBuffer(base64) {
  const base64Data = base64.replace(/^data:image\/\w+;base64,/, "");
  return Buffer.from(base64Data, "base64");
}

// Convert buffer to Image
async function bufferToImage(buffer) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = buffer;
  });
}

// Compare faces
async function compareFaces(img1Base64, img2Base64) {
  const img1Buffer = base64ToBuffer(img1Base64);
  const img2Buffer = base64ToBuffer(img2Base64);

  const img1 = await bufferToImage(img1Buffer);
  const img2 = await bufferToImage(img2Buffer);

  const face1 = await faceapi
    .detectSingleFace(img1, new faceapi.MtcnnOptions()) // Use MTCNN options
    .withFaceLandmarks()
    .withFaceDescriptor();
  const face2 = await faceapi
    .detectSingleFace(img2, new faceapi.MtcnnOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!face1 || !face2) {
    throw new Error("Faces not detected in one or both images");
  }

  const distance = faceapi.euclideanDistance(
    face1.descriptor,
    face2.descriptor
  );
  const similarity = (1 - distance) * 100;

  // Lower threshold means stricter verification
  const threshold = 0.4;

  return {
    similarity: similarity.toFixed(2) + "%",
    distance: distance.toFixed(4),
    threshold: threshold,
    isVerified: distance < threshold,
  };
}

app.post("/compare", async (req, res) => {
  try {
    const { image1, image2 } = req.body;

    if (!image1 || !image2) {
      return res.status(400).json({ error: "Both images are required" });
    }

    const result = await compareFaces(image1, image2);

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/", (req, res) => {
  return res.json({ message: "Hi Node" });
});

app.listen(port, async () => {
  await loadModels();
  console.log(`Face comparison API running on http://localhost:${port}`);
});
