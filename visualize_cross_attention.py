import torch
from torch import nn
from PIL import Image
from layers import TokenCreation
import matplotlib.pyplot as plt
from torchvision import transforms
from timm.models.vision_transformer import PatchEmbed
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class TokenPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
                img_size=224,
                patch_size=4,
                in_chans=3,
                embed_dim=384,
            )
        self.token_creation = TokenCreation(embed_dim=384, num_tokens=196)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, 384))

    def forward(self, x):
        B, w, h, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed #self.interpolate_pos_encoding(x, w, h)  

        attn = self.token_creation(x, return_attn=True)
        
        return attn


def main():
    # State dict loading
    state_dict = torch.load(
            "/home/3147347/checkpoint/experiments/66417/best_checkpoint.pth",
            map_location="cpu"
        )["model"]
    
    # Initialization of Token Preprocessing
    token_preprocess = TokenPreprocess()
    token_preprocess.load_state_dict(state_dict, strict=False)

    # Image transformation
    transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    
    # Image loading
    img_path = "./image.JPEG"
    img = Image.open(img_path)
    print(transform(img).shape)

    attn = token_preprocess(transform(img))

    print(attn.shape)


if __name__ == "__main__":
    main()
