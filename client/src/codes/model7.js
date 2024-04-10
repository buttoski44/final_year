export const model7 = {
  fileName: "Training.js",
  code: `
    Dense_args = AttrDict()
    ## Args setting
    Dense_args_dict = {
        'model': Dense,
        'num_epochs': 5,
        'batch_size': 16,
        'criterion': criterion_Dense,
        'optimizer': optimizer_Dense,
        'scheduler': exp_lr_scheduler_Dense,
        'save_checkpoints': True,
        'experiment_name': "Dense"
    }
    ## Training process
    Dense_args.update(Dense_args_dict)
    Dense_output_model = train_model(Dense_args)
    EfficientNet_B0_args = AttrDict()
    ## Args setting
    EfficientNet_B0_args_dict = {
        'model': EfficientNet_B0,
        'num_epochs': 5,
        'batch_size': 16,
        'criterion': criterion_EfficientNet_B0,
        'optimizer': optimizer_EfficientNet_B0,
        'scheduler': exp_lr_scheduler_EfficientNet_B0,
        'save_checkpoints': True,
        'experiment_name': "EfficientNet_B0"
    }
    MobileNet_V3_args = AttrDict()
    ## Args setting
    MobileNet_V3_args_dict = {
        'model': MobileNet_V3,
        'num_epochs': 5,
        'batch_size': 16,
        'criterion': criterion_MobileNet_V3,
        'optimizer': optimizer_MobileNet_V3,
        'scheduler': exp_lr_scheduler_MobileNet_V3,
        'save_checkpoints': True,
        'experiment_name': "MobileNet_V3"
    }
    ## Training process
    MobileNet_V3_args.update(MobileNet_V3_args_dict)
    MobileNet_V3_output_model = train_model(MobileNet_V3_args)
    ## Training process
    EfficientNet_B0_args.update(EfficientNet_B0_args_dict)
    EfficientNet_B0_output_model = train_model(EfficientNet_B0_args)
    def train_model(args):
    """
    Train the Neural Networks
    Args:
        args, dictionary containing all the arguments used in training
            args.model: class name of the model
            args.num_epochs(int): number of epochs
            args.batch_size(int): batch size
            args.criterion: criterion used in training
            args.optimizer: optimizer used in training
            args.scheduler: scheduler used in training
            args.save_checkpoints(bool): save checkpoints
            args.experiment_name(str): experiment name, used as folder name
        net, neural net to be trained
    Return:
        The best neural net in the training
    """
    start_time = time.time()
        
    save_dir = "output/" + args.experiment_name + "/"
        
    # if path does not exist, create one
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    best_weight = copy.deepcopy(args.model.state_dict())
    best_acc = 0.0
        
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
        
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch}/{args.num_epochs - 1}', end=': ')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                args.model.train() # training mode
            else:
                args.model.eval()  # evaluate mode
        
            running_loss = 0.0
            running_corrects = 0
        
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                # zero the parameter gradients
                args.optimizer.zero_grad()
        
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = args.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = args.criterion(outputs, labels)
        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        args.optimizer.step()
        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
            print(f'{phase} loss: {epoch_loss:.4f} {phase} acc: {epoch_acc:.4f}', end=', ')
        
            if phase == 'train':
                args.scheduler.step()
                train_accs.append(epoch_acc.cpu().detach())
                train_losses.append(epoch_loss)
            else:
                val_accs.append(epoch_acc.cpu().detach())
                val_losses.append(epoch_loss)
        
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(args.model.state_dict())
            
            # save checkpoints every epoch
            if args.save_checkpoints:
                checkpoints_dir = save_dir 
                    + f"model_{args.experiment_name}_bs_{args.batch_size}_epoch_{epoch}"
                torch.save(args.model.state_dict(), checkpoints_dir)
        
        epoch_dur = round(time.time() - epoch_start,2)
        print(f'Epoch time:  {epoch_dur // 60:.0f}m {epoch_dur % 60:.0f}s')
        
    time_elapsed = time.time() - start_time
    print()
    print('-' * 20)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
        
    # save accuracy and loss
    if args.save_checkpoints:
        np.savetxt(save_dir + f"model_{args.experiment_name}_train_acc.csv", train_accs)
        np.savetxt(save_dir 
            + f"model_{args.experiment_name}_train_loss.csv", train_losses)
        np.savetxt(save_dir + f"model_{args.experiment_name}_val_acc.csv", val_accs)
        np.savetxt(save_dir + f"model_{args.experiment_name}_val_loss.csv", val_losses)
        
    # plot traininng curve
    train_accs, val_accs = np.array(train_accs), np.array(val_accs)
    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
        
    plt.plot(np.arange(args.num_epochs, step=1), train_losses, label='Train loss')
    plt.plot(np.arange(args.num_epochs, step=1), train_accs, label='Train acc')
    plt.plot(np.arange(args.num_epochs, step=1), val_accs, label='Val acc')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
        
    # load and save the best model weights
    checkpoints_dir = save_dir 
        + f"model_{args.experiment_name}_bs_{args.batch_size}_best_model"
    torch.save(args.model.state_dict(best_model_wts), checkpoints_dir)
    args.model.load_state_dict(best_model_wts)
        
    return args.model
                `,
};
