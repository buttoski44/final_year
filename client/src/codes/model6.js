export const model6 = {
  fileName: "Mobile Net.js",
  code: `
    from torchinfo import summary

    MobileNet_V3 = torchvision.models.mobilenet_v3_large(weights='DEFAULT').to(device)
    
    # Freeze all base layers in the "features" section of the model 
    (the feature extractor) by setting requires_grad=False
    for param in MobileNet_V3.features.parameters():
        param.requires_grad = False
    
    # Recreate the classifier layer and seed it to the target device
    MobileNet_V3.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=960, 
                        out_features=len(class_names), 
                        # same number of output units as our number of classes
                        bias=True)).to(device)
    
    ## Model inItialization
    criterion_MobileNet_V3 = nn.CrossEntropyLoss()
    optimizer_MobileNet_V3 = optim.SGD(
        MobileNet_V3.parameters(), 
        lr=0.005, 
        momentum=0.9
    )
    exp_lr_scheduler_MobileNet_V3 = lr_scheduler.StepLR(
        optimizer_MobileNet_V3, 
        step_size=7, 
        gamma=0.1
    )
              `,
};
