# %%
# Library

# %%
def sample(args):
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    diffusion.load_state_dict(torch.load(args.model_path))

    if args.use_labels:
        for label in range(10):
            y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
            samples = diffusion.sample(args.num_images // 10, device, y=y)

            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
    else:
        samples = diffusion.sample(args.num_images, device)

        for image_id in range(len(samples)):
            image = ((samples[image_id] + 1) / 2).clip(0, 1)
            torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")