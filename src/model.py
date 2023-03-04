import torch
import torch.nn as nn


class PatchEmbed(nn.Module): 
    """
    Split image into patches and then embed them
    
    Parameters: 
        img_size: Size of the image (assuming it is a square)
        patch_size: Size of the patch (asuming it is a square)
        in_chans: Number of input channels
        embed_dim: The embedding dimensions

    Attrs:
        n_patches: Number of patches inside of our image
        proj: conv layer that does both the splitting into patches and their embeddings
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768):
        super().__init()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

