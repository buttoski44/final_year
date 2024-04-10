export const model3 = {
  fileName: "Agumentation.py",
  code: `
    # Calculate means and stds of the trainset and normalize

    train_data = torchvision.datasets.ImageFolder(
        root = output_dir+'/train', 
        transform = transforms.ToTensor()
    )
    
    means = torch.zeros(3)
    stds = torch.zeros(3)
    
    for img, label in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))
    
    means /= len(train_data)
    stds /= len(train_data)
        
    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')
    
    data_augmentation = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            torchvision.transforms.AutoAugment(
                torchvision.transforms.AutoAugmentPolicy.IMAGENET
                ),
            transforms.ToTensor(),
            #transforms.Normalize(mean = means, std = stds),
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            torchvision.transforms.AutoAugment(
                torchvision.transforms.AutoAugmentPolicy.IMAGENET
                ),
            transforms.ToTensor(),
            #transforms.Normalize(mean = means, std = stds),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'cust_test': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
            `,
};
