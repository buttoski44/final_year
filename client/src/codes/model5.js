export const model5 = {
  fileName: "Efficient Net.js",
  code: `
    from torchinfo import summary
    
    EfficientNet_B0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT 
    EfficientNet_B0 = torchvision.models.efficientnet_b0(
      weights=EfficientNet_B0_weights).to(device)
    
    # Freeze all base layers in the "features" section of the 
    model (the feature extractor) by setting requires_grad=False
    for param in EfficientNet_B0.features.parameters():
        param.requires_grad = False
    
    # Recreate the classifier layer and seed it to the target device
    EfficientNet_B0.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, out_features=len(class_names))).to(device)
        
    ## Model inItialization
    criterion_EfficientNet_B0 = nn.CrossEntropyLoss()
    optimizer_EfficientNet_B0 = optim.SGD(
      EfficientNet_B0.parameters(), 
      lr=0.005, 
      momentum=0.9)
    exp_lr_scheduler_EfficientNet_B0 = lr_scheduler.StepLR(
      optimizer_EfficientNet_B0, 
      step_size=7, 
      gamma=0.1)
            `,
};
