import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
from featup.util import norm, unnorm, pca, remove_axes
from pytorch_lightning import seed_everything
import os, requests, csv

def plot_feats(image, lr, hr):
    # return plt figure
    plt.clf()
    plt.cla()
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)], dim=9)
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    ax[0, 0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[1, 0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[2, 0].imshow(image.permute(1, 2, 0).detach().cpu())

    ax[0, 0].set_title("Image", fontsize=22)
    ax[0, 1].set_title("Original", fontsize=22)
    ax[0, 2].set_title("Upsampled Features", fontsize=22)

    ax[0, 1].imshow(lr_feats_pca[0, :3].permute(1, 2, 0).detach().cpu())
    ax[0, 0].set_ylabel("PCA Components 1-3", fontsize=22)
    ax[0, 2].imshow(hr_feats_pca[0, :3].permute(1, 2, 0).detach().cpu())

    ax[1, 1].imshow(lr_feats_pca[0, 3:6].permute(1, 2, 0).detach().cpu())
    ax[1, 0].set_ylabel("PCA Components 4-6", fontsize=22)
    ax[1, 2].imshow(hr_feats_pca[0, 3:6].permute(1, 2, 0).detach().cpu())

    ax[2, 1].imshow(lr_feats_pca[0, 6:9].permute(1, 2, 0).detach().cpu())
    ax[2, 0].set_ylabel("PCA Components 7-9", fontsize=22)
    ax[2, 2].imshow(hr_feats_pca[0, 6:9].permute(1, 2, 0).detach().cpu())

    remove_axes(ax)
    plt.tight_layout()    
    return fig


if __name__ == "__main__":

    def download_image(url, save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as file:
            file.write(response.content)

    base_url = "https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/sample_images/"
    sample_images_urls = {
        "skate.jpg": base_url + "skate.jpg",
        "car.jpg": base_url + "car.jpg",
        "plant.png": base_url + "plant.png",
    }

    local_images_dir = "/tmp/sample_images"

    # Ensure the directory for sample images exists
    os.makedirs(local_images_dir, exist_ok=True)

    # Download each sample image
    for filename, url in sample_images_urls.items():
        save_path = os.path.join(local_images_dir, filename)
        # Download the image if it doesn't already exist
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            download_image(url, save_path)
        else:
            print(f"{filename} already exists. Skipping download.")

    os.environ['TORCH_HOME'] = '/tmp/.cache'
    os.environ['GRADIO_EXAMPLES_CACHE'] = '/tmp/gradio_cache'
    csv.field_size_limit(100000000)
    options = ['dino16', 'vit', 'dinov2', 'clip', 'resnet50']
    models = []
    models_name = []
    for o in options:
        print(f'About to load {o}')
        try:
            model = torch.hub.load("mhamilton723/FeatUp", o)
            print(f'...SUCCESS')
            models.append(model)
            models_name.append(o)
        except Exception as e:
            print(f'...ERROR. Skipping')

    def upsample_features(image, model_option:int):
        # Image preprocessing
        input_size = 224
        transform = T.Compose([
            T.Resize(input_size),
            T.CenterCrop((input_size, input_size)),
            T.ToTensor(),
            norm
        ])
        image_tensor = transform(image).unsqueeze(0).cuda()
        # Load the selected model
        upsampler = models[model_option].cuda()
        hr_feats = upsampler(image_tensor)
        lr_feats = upsampler.model(image_tensor)
        upsampler.cpu()
        return plot_feats(unnorm(image_tensor)[0], lr_feats[0], hr_feats[0])
    #
    for j, img_fn in enumerate(['skate.jpg', 'car.jpg', 'plant.png']):
        img_fn_abs = os.path.join(local_images_dir, img_fn)
        if os.path.exists(img_fn_abs):
            image = Image.open(img_fn_abs) # torchvision.io.read_image(img_fn)
            for i, model_option in enumerate(models_name): 
                fig = upsample_features(image, model_option=i)
                fname = f'{img_fn}-ALGO={model_option}.png'
                plt.savefig(fname)
                plt.close(fig)  # Close plt to avoid additional empty plots
                print(f'{fname} generated with...SUCCESS')
