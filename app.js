const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cors = require("cors");
const app = express();
const port = 3000;
// Wrap the code in an asynchronous function
(async () => {
  // Load the pre-trained model
  const modelPath =
    "file:///Users/vishal.sangole/Downloads/CONGONITIVE-DRIVER-ACTION-RECOGNITION-SYSTEM-main/driverdistraction_tfjs/model.json";
  const model = await tf.loadLayersModel(modelPath);
  const storage = multer.memoryStorage();
  const upload = multer({ storage: storage });
  app.use(cors());
  app.post("/classify", upload.single("image"), async (req, res) => {
    console.log("ivsajbslbdsljbsljbaslbsaljbskljs");
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const imageBuffer = req.file.buffer;
      // Decode the image from base64
      const imageArray = new Uint8Array(imageBuffer);
      const imageTensor = tf.node.decodeImage(imageArray);

      //const preprocessedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
      const preprocessedImage = tf.image.resizeBilinear(
        imageTensor,
        [240, 240]
      );

      const inputTensor = preprocessedImage.expandDims();

      // Make predictions using predict
      const predictions = model.predict(inputTensor);
      //console.log(`Shape of model 's output: ${predictions.shape}`);
      // Get the predicted class
      const predictedClass = tf.argMax(predictions, 1).dataSync()[0];

      res.json({ class: predictedClass });
    } catch (error) {
      console.log(res);
      // console.error(error.message);
      // res.status(500).json({ error: "Internal Server Error" });
    }
  });

  // Start the server
  app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  });
})();
