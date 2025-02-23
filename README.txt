move node_modules\@tensorflow\tfjs-node\deps\lib\tensorflow.dll to node_modules\@tensorflow\tfjs-node\lib\napi-v6\




BACKUPP


const express = require("express");
const faceapi = require("face-api.js");
const tf = require("@tensorflow/tfjs-node");
const canvas = require("canvas");
const path = require("path");

// Initialize canvas for face-api.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const port = 3000;

app.use(express.json({ limit: "10mb" }));

// Load models
const MODEL_URL = path.join(__dirname, "models");

async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
}

// Calculate face similarity
async function compareFaces(img1Base64, img2Base64) {
  const img1 = await canvas.loadImage(`data:image/jpeg;base64,${img1Base64}`);
  const img2 = await canvas.loadImage(`data:image/jpeg;base64,${img2Base64}`);

  const face1 = await faceapi
    .detectSingleFace(img1)
    .withFaceLandmarks()
    .withFaceDescriptor();
  const face2 = await faceapi
    .detectSingleFace(img2)
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

  // Adjust threshold for better accuracy
  const threshold = 0.4; // Lowered for stricter face verification

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

app.get("/", async (req, res) => {
  return res.json({ message: "Hi Node" });
});

app.listen(port, async () => {
  await loadModels();
  console.log(`Face comparison API running on http://localhost:${port}`);
});
