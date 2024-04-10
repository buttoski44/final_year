export const model8 = {
  fileName: "Ensambler.js",
  code: `
    class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.num_models = len(models)
        self.models = models
        self.weights = torch.ones(self.num_models,10)
        
    ensemble3 = EnsembleModel([Dense, MobileNet_V3, EfficientNet_B0])
            `,
};
