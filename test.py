import os
import yaml
import torch
from argparse import ArgumentParser
from model import my_net
from tools.my_utils import img_loader, apply_super_resolution, apply_denoising, ensure_dir
from torchvision import transforms
from PIL import Image

if __name__ == "__main__":
    # Load configuration
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=config.get('test_input_dir', './data/UDD/test'))
    parser.add_argument("--output_dir", type=str, default=config.get('test_output_dir', './test'))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(config.get('save_dir', './model/checkpoints'), config.get('best_model_name', 'best_SR.pth')))
    parser.add_argument("--device", type=str, default=config.get('device', 'auto'))
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"#==> Using {'GPU' if device != 'cpu' else 'CPU'} device")

    # Model setup
    channel_in = 1 if config['input_type'] == 'L' else 3
    channel_out = 1 if config['output_type'] == 'L' else 3
    factor = int(config['factor'])
    in_patch_size = int(config['patch_size'] // factor)
    model = my_net(channel_in, channel_out, factor).to(device)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"#==> Loading model checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("#==> Model loaded successfully")
    else:
        print(f"#==> WARNING: Checkpoint {args.checkpoint} not found. Running with randomly initialized model.")

    model.eval()

    # Prepare output directory
    ensure_dir(args.output_dir)

    # List input images
    input_images = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
    if not input_images:
        print(f"#==> No images found in {args.input_dir}")
        exit(1)

    for img_name in input_images:
        img_path = os.path.join(args.input_dir, img_name)
        image = img_loader(img_path, config['input_type'])
        w, h = image.size
        img_output_size = (w * factor, h * factor)

        # Super-resolution
        sr_tensor = apply_super_resolution(
            image_lr_in=image,
            model=model,
            channel_out=channel_out,
            in_patch_size=in_patch_size,
            img_output_size=img_output_size,
            factor=factor,
            device=device
        )
        sr_img = transforms.ToPILImage()(sr_tensor.cpu().clamp(0, 1))
        sr_img.save(os.path.join(args.output_dir, f'sr_{os.path.splitext(img_name)[0]}.png'))

        # Denoising
        denoised_tensor = apply_denoising(
            image_lr_in=image,
            model=model,
            channel_out=channel_out,
            in_patch_size=in_patch_size,
            factor=factor,
            device=device
        )
        denoised_img = transforms.ToPILImage()(denoised_tensor.cpu().clamp(0, 1))
        denoised_img.save(os.path.join(args.output_dir, f'denoised_{os.path.splitext(img_name)[0]}.png'))

        print(f"#==> Processed {img_name}: saved super-resolved and denoised outputs.")

    print(f"#==> All images processed. Results saved to {args.output_dir}")
