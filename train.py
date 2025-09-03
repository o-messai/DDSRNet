from tensorboardX import SummaryWriter
from torchsummary import summary
from datetime import datetime
from argparse import ArgumentParser
from model import my_net
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from tools.my_load_dataset import load_hr_and_lr
from torchmetrics.image import PeakSignalNoiseRatio
import numpy as np
import yaml
import shutil
import os
from PIL import Image
from tools.my_utils import ensure_dir
from tools.my_loss import SSIM_Loss, CompositeLoss


if __name__ == '__main__':
    # Load configuration from config.yaml
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=config.get('batch_size', 64))
    parser.add_argument("--epochs", type=int, default=config.get('epochs', 100))
    parser.add_argument("--lr", type=float, default=config.get('learning_rate', 0.0001))
    parser.add_argument("--dataset_name", type=str, default=config.get('dataset_name', "udd_R1T1"))
    parser.add_argument("--weight_decay", type=float, default=config.get('weight_decay', 0.001))
    parser.add_argument("--lambda_weight", type=float, default=config.get('lambda_weight', 1.0), help="Weight for denoising loss")
    parser.add_argument("--beta_weight", type=float, default=config.get('beta_weight', 1.0), help="Weight for super-resolution loss")
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['dataset_name'] = args.dataset_name
    config['weight_decay'] = args.weight_decay
    config['lambda_weight'] = args.lambda_weight
    config['beta_weight'] = args.beta_weight

    log_file_path = os.path.join(config.get('log_dir', './results'), 'performance_logs.txt')
    ensure_dir(config.get('log_dir', 'results'))
    
    with open(log_file_path, 'a+') as f:
        now = datetime.now()
        f.write(f"\n\n #============================== Experiment date and time = {now}.==============================#")
        f.write(f"\n dataset = {config['dataset_name']} epochs = {config['epochs']}, batch_size = {config['batch_size']}, lr = {config['learning_rate']}, weight_decay= {config['weight_decay']}")
        f.write(f"\n lambda_weight = {config['lambda_weight']}, beta_weight = {config['beta_weight']}")
        f.write(f"\n DYNAMIC LOSS STRATEGY: λ starts at 1.0, β starts at 0.0")
        f.write(f"\n After every 10 epochs: λ decreases by 0.1, β increases by 0.1")
        f.write(f"\n At epoch 50+: λ=β=0.5 (balanced training)")
        f.write(f"\n {config}")
        seed = config.get('seed', 14732152)
        torch.manual_seed(seed)
        np.random.seed(seed)
        f.write(f"\n Seed : {seed}")

    device = torch.device('cuda:0' if torch.cuda.is_available() and config.get('device', 'auto') != 'cpu' else 'cpu')
    print(f'#==> Using {"GPU" if device != "cpu" else "CPU"} device')
    
    # Print dynamic loss strategy information
    dynamic_loss_msg = (
        "#==> DYNAMIC LOSS STRATEGY:\n"
        "#==> Epochs 0-9:   λ=1.0, β=0.0 (Pure denoising focus)\n"
        "#==> Epochs 10-19: λ=0.9, β=0.1 (Start introducing super-resolution)\n"
        "#==> Epochs 20-29: λ=0.8, β=0.2 (Gradually increase SR focus)\n"
        "#==> Epochs 30-39: λ=0.7, β=0.3 (Continue transition)\n"
        "#==> Epochs 40-49: λ=0.6, β=0.4 (Near balanced)\n"
        "#==> Epochs 50+:   λ=0.5, β=0.5 (Balanced training)\n"
        "#==> This strategy allows optimized training of both denoising and super-resolution tasks"
    )
    print(dynamic_loss_msg)

    ensure_dir(config.get('save_dir', './model'))
    save_model = os.path.join(config.get('save_dir', './model/checkpoints'), config.get('best_model_name', 'best_SR.pth'))
    shutil.rmtree('./tensorboard', ignore_errors=True)
    ensure_dir(config.get('tensorboard_dir', './tensorboard'))
    writer = SummaryWriter(config.get('tensorboard_dir', './tensorboard'))
    
    test_dataset = load_hr_and_lr(config['dataset_name'], config, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)
    train_dataset = load_hr_and_lr(config['dataset_name'], config, "train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=config.get('shuffle', True), 
        pin_memory=config.get('pin_memory', True), 
        num_workers=config.get('num_workers', 0)
    )

    factor = int(config['factor'])
    input_size = int(config['patch_size']/factor)
    channel_in = 1 if config['input_type'] == 'L' else 3
    channel_out = 1 if config['output_type'] == 'L' else 3
    
    # Validate patch size is divisible by factor
    if config['patch_size'] % factor != 0:
        adjusted_patch_size = (config['patch_size'] // factor) * factor
        print(f"#==> WARNING: Patch size {config['patch_size']} not divisible by factor {factor}")
        print(f"#==> Adjusting patch size to {adjusted_patch_size}")
        config['patch_size'] = adjusted_patch_size
        input_size = int(config['patch_size']/factor)

    model = my_net(channel_in, channel_out).to(device)
    summary(model, input_size=[(channel_in, input_size, input_size)])
    
    # Import CombinedLoss for dynamic loss strategy
    from tools.my_loss import CombinedLoss
    
    # Dynamic loss strategy: Start with denoising focus, gradually transition to balanced training
    # Initial values: λ=1.0, β=0.0 (pure denoising)
    # After every 10 epochs: increase β by 0.1, decrease λ by 0.1
    # Stop at epoch 50 where λ=β=0.5 (balanced training)
    # This method is better than static loss as it allows optimized training of both tasks
    def get_dynamic_weights(epoch):
        if epoch < 10:
            return 1.0, 0.0  # λ=1.0, β=0.0 (epochs 0-9): Pure denoising focus
        elif epoch < 20:
            return 0.9, 0.1  # λ=0.9, β=0.1 (epochs 10-19): Start introducing super-resolution
        elif epoch < 30:
            return 0.8, 0.2  # λ=0.8, β=0.2 (epochs 20-29): Gradually increase SR focus
        elif epoch < 40:
            return 0.7, 0.3  # λ=0.7, β=0.3 (epochs 30-39): Continue transition
        elif epoch < 50:
            return 0.6, 0.4  # λ=0.6, β=0.4 (epochs 40-49): Near balanced
        else:
            return 0.5, 0.5  # λ=0.5, β=0.5 (epochs 50+): Balanced training
    
    # Create the combined loss criterion that will be updated with dynamic weights
    criterion = CombinedLoss(
        lambda_weight=1.0,  # Initial value, will be updated dynamically
        beta_weight=0.0,    # Initial value, will be updated dynamically
        data_range=config.get('data_range', 1.0), 
        size_average=config.get('size_average', True), 
        channel=config.get('channel', channel_out)
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('scheduler_step', 50), 
        gamma=config.get('scheduler_gamma', 0.8)
    )

    # Initialize best metrics for model saving
    best_loss = float('inf')  # Track best validation loss (lower is better)
    best_epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(config.get('save_dir', './model/checkpoints'))
    ensure_dir(checkpoint_dir)
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    # Check if there's a checkpoint to resume from
    start_epoch = 0
    
    if os.path.exists(latest_checkpoint_path) and config.get('resume_training', False):
        print(f"#==> Loading checkpoint from {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        print(f"#==> Resumed training from epoch {start_epoch}")
        print(f"#==> Best loss so far: {best_loss:.6f} at epoch {best_epoch}")
    
    # Load best model if it exists
    if os.path.exists(save_model):
        print(f"#==> Loading best model from {save_model}")
        model.load_state_dict(torch.load(save_model, map_location=device))
        print(f"#==> Best model loaded successfully")
    
    for epoch in tqdm(range(start_epoch, config['epochs']), 'training epochs'):
        # Update dynamic weights for this epoch
        lambda_weight, beta_weight = get_dynamic_weights(epoch)
        criterion.lambda_weight = lambda_weight
        criterion.beta_weight = beta_weight
        
        # Log current weights
        print(f'#==> Epoch {epoch}: λ={lambda_weight:.1f}, β={beta_weight:.1f}')
        writer.add_scalar('train/lambda_weight', lambda_weight, epoch)
        writer.add_scalar('train/beta_weight', beta_weight, epoch)
        
        LOSS_all = 0
        DENOISING_LOSS_all = 0
        SR_LOSS_all = 0
        
        for batch_idx, (patchesHR, patchesLR, patchesLR_nn) in enumerate(train_loader):
            patchesHR, patchesLR, patchesLR_nn = patchesHR.to(device), patchesLR.to(device), patchesLR_nn.to(device)
            optimizer.zero_grad()
            
            # Forward pass for both denoising and super-resolution
            pred_HR, pred_LR_denoised = model(patchesLR)  # Model returns (img_super, img_denoised)
            
            # Combined loss: λ * D_denoising + β * D_super_resolution
            total_loss, denoising_loss, sr_loss = criterion(
                denoising_ref=patchesLR_nn,      # Reference: clean LR
                denoising_out=pred_LR_denoised,  # Output: denoised LR
                sr_ref=patchesHR,                # Reference: high-resolution
                sr_out=pred_HR                   # Output: super-resolved
            )
            
            total_loss.backward()
            optimizer.step()
            
            LOSS_all += total_loss.item()
            DENOISING_LOSS_all += denoising_loss.item()
            SR_LOSS_all += sr_loss.item()

        LOSS_all /= (batch_idx + 1)
        DENOISING_LOSS_all /= (batch_idx + 1)
        SR_LOSS_all /= (batch_idx + 1)
        
        print(f'#==> Training_loss total: {LOSS_all:.6f}, denoising: {DENOISING_LOSS_all:.6f}, super_resolution: {SR_LOSS_all:.6f}')
        writer.add_scalar('train/loss_total', LOSS_all, epoch)
        writer.add_scalar('train/denoising_loss', DENOISING_LOSS_all, epoch)
        writer.add_scalar('train/super_resolution_loss', SR_LOSS_all, epoch)
        
        # Step the scheduler
        scheduler.step()

        model.eval()
        LOSS_all = 0
        DENOISING_LOSS_all = 0
        SR_LOSS_all = 0
        best_SSIM = float('inf')
        
        with torch.no_grad():
            for batch_idx, (patchesHR, patchesLR, patchesLR_nn) in enumerate(test_loader):
                patchesLR, patchesHR, patchesLR_nn = patchesLR.to(device), patchesHR.to(device), patchesLR_nn.to(device)
                
                # Forward pass for both tasks
                pred_HR, pred_LR_denoised = model(patchesLR)  # Model returns (img_super, img_denoised)
                
                # Log images to TensorBoard (only for the first batch to avoid flooding)
                if batch_idx == 0:
                    # Take the first image in the batch
                    input_img = patchesLR[0].detach().cpu()
                    denoised_img = pred_LR_denoised[0].detach().cpu()
                    sr_img = pred_HR[0].detach().cpu()
                    gt_img = patchesHR[0].detach().cpu()
                    
                    # Normalize images to [0, 1] if needed
                    def norm_img(img):
                        return (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    writer.add_image('test/input_LR', norm_img(input_img), epoch)
                    writer.add_image('test/denoised_LR', norm_img(denoised_img), epoch)
                    writer.add_image('test/super_resolved', norm_img(sr_img), epoch)
                    writer.add_image('test/ground_truth_HR', norm_img(gt_img), epoch)
                
                # Test loss using current dynamic weights
                total_loss, denoising_loss, sr_loss = criterion(
                    denoising_ref=patchesLR_nn,
                    denoising_out=pred_LR_denoised,
                    sr_ref=patchesHR,
                    sr_out=pred_HR
                )
                
                LOSS_all += total_loss.item()
                DENOISING_LOSS_all += denoising_loss.item()
                SR_LOSS_all += sr_loss.item()

            LOSS_all /= (batch_idx + 1)
            DENOISING_LOSS_all /= (batch_idx + 1)
            SR_LOSS_all /= (batch_idx + 1)
            
            print(f'#==> Test dataset patches loss - total: {LOSS_all:.6f}, denoising: {DENOISING_LOSS_all:.6f}, super_resolution: {SR_LOSS_all:.6f}')
            writer.add_scalar('test_patches/loss_total', LOSS_all, epoch)
            writer.add_scalar('test_patches/denoising_loss', DENOISING_LOSS_all, epoch)
            writer.add_scalar('test_patches/super_resolution_loss', SR_LOSS_all, epoch)

            if LOSS_all < best_SSIM:
                best_SSIM = LOSS_all
                best_epoch = epoch
                torch.save(model.state_dict(), save_model)
                print(f"#==> New best model saved at epoch {epoch} with loss {best_SSIM:.6f}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(config.get('save_dir', './model/checkpoints'), f'epoch_{epoch}_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_SSIM, # Use best_SSIM as best_loss
                'best_epoch': best_epoch, # Store the epoch when best_SSIM was achieved
                'config': config
            }, checkpoint_path)
            print(f"#==> Checkpoint saved at {checkpoint_path}")

        # Test on input images
        test_input_dir = config.get('test_input_dir', './results/input_img')
        test_output_dir = config.get('test_output_dir', './test')
        
        if os.path.exists(test_input_dir):
            ensure_dir(test_output_dir)
            for img_name in os.listdir(test_input_dir):
                img_path = os.path.join(test_input_dir, img_name)
                image = Image.open(img_path).convert('L').crop((0, 0, input_size, input_size))
                image = transforms.ToTensor()(image).unsqueeze(0).to(device)
                
                # Get both super-resolution and denoising outputs
                pred_HR, pred_LR_denoised = model(image)
                
                # Save super-resolution output
                output_image = transforms.ToPILImage()(pred_HR.squeeze(0).cpu())
                output_image.save(os.path.join(test_output_dir, f'sr_{img_name[:-4]}.png'))
                
                # Save denoising output
                output_image = transforms.ToPILImage()(pred_LR_denoised.squeeze(0).cpu())
                output_image.save(os.path.join(test_output_dir, f'denoised_{img_name[:-4]}.png'))
    
    # Training completed - print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED - SUMMARY")
    print("="*80)
    print(f"Best model saved at: {save_model}")
    print(f"Best validation loss: {best_SSIM:.6f} at epoch {best_epoch}") # Use best_SSIM as best_loss
    print(f"Latest checkpoint saved at: {checkpoint_path}")
    print(f"Total epochs trained: {config['epochs']}")
    print(f"Final learning rate: {scheduler.get_last_lr()[0]:.6f}")
    print("="*80)
    
    # Log final results
    with open(log_file_path, 'a+') as f:
        f.write(f"\n\n #============================== Training Completed ==============================#")
        f.write(f"\n Best validation loss: {best_SSIM:.6f} at epoch {best_epoch}") # Use best_SSIM as best_loss
        f.write(f"\n Best model saved at: {save_model}")
        f.write(f"\n Latest checkpoint: {checkpoint_path}")
        f.write(f"\n Final learning rate: {scheduler.get_last_lr()[0]:.6f}")
        f.write(f"\n #============================== End of Training ==============================#")
    
    writer.close()
    print(f"#==> Training completed. Best model saved at {save_model}")
    print(f"#==> Checkpoint saved at {checkpoint_path}")
    print(f"#==> TensorBoard logs saved at {config.get('tensorboard_dir', './tensorboard')}")
    print(f"#==> Performance logs saved at {log_file_path}")