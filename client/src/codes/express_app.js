export const code3 = {
  fileName: "Express_App.js",
  code: `
    const app = express();
    const port = 3000;
  
    (async () => {
        const modelPath = 'api/model/driverdistraction_tfjs/model.json';
        const model = await tf.loadLayersModel(modelPath);
        const storage = multer.memoryStorage();
        const upload = multer({ storage: storage });
    // ...
  })();
        `,
};
