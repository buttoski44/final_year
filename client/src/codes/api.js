export const code4 = {
  fileName: "Post_Request.js",
  code: `
  app.post('/classify', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
          return res.status(400).json({ error: 'No file uploaded' });
        }
        const imageBuffer = req.file.buffer

        // Decode the image from base64
        const imageArray = new Uint8Array(imageBuffer);
        const imageTensor = tf.node.decodeImage(imageArray);

        //const preprocessedImage = tf.image.resizeBilinear(
          imageTensor, [224, 224]
          );
        const preprocessedImage = tf.image.resizeBilinear(
          imageTensor, [240, 240]
          );
        const inputTensor = preprocessedImage.expandDims();

        // Make predictions using predict
        const predictions = model.predict(inputTensor);

        // Get the predicted class
        const predictedClass = tf.argMax(predictions, 1).dataSync()[0];
        res.json({ class: predictedClass });
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' });
    }
  });`,
};
