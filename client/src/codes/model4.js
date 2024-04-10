export const model4 = {
  fileName: "Desne Net.js",
  code: `
    from torchinfo import summary

    Dense = torchvision.models.densenet121(weights='DEFAULT').to(device)
    
    # Freeze all base layers in the "features" section of the 
    model (the feature extractor) by setting requires_grad=False
    for param in Dense.features.parameters():
        param.requires_grad = False
    
    # Recreate the classifier layer and seed it to the target device
    Dense.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1024, 
                        out_features=len(class_names), 
                        # same number of output units as our number of classes
                        bias=True)).to(device)

    ## Model inItialization
    criterion_Dense = nn.CrossEntropyLoss()
    optimizer_Dense = optim.SGD(Dense.parameters(), lr=0.005, momentum=0.9)
    exp_lr_scheduler_Dense = lr_scheduler.StepLR(optimizer_Dense, step_size=7, gamma=0.1)
          `,
};
