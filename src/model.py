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
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: torch.Tensor (n_samples, in_channels, width, height)
        :return: torch.Tensor (n_samples, n_patches, embed_dim)
        """

        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches
        x = x.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0, proj_p: float = 0):
        """
        :param dim: The input and out dimension of per token features
        :param n_heads:Number of attention heads
        :param qkv_bias: If True then we include bias
        :param attn_p: attention dropout applied to qkv
        :param proj_p: dropout applied to the output tensor
        """
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads  # dimensionality of each head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

        def forward(self, x):
            """
            :param x: torch.Tensor (n_samples, n_patched + 1, dim)
            :return: torch.Tensor (n_samples, n_patched + 1, dim)
            """
            n_samples, n_tokens, dim = x.shape

            if dim != self.dim:
                raise ValueError

            qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3*dim)
            qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches+1, head_dim)

            q, k, v = qkv[0], qkv[1], qkv[2]  # extract k, q, v, vectors
            k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches+1)
            dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches+1, n_patches+1)
            attn = dp.softmax(dim=-1)
            attn = self.attn_drop(attn)

            weighted_avg = attn @ v  # (n_samples, n_heads, n_patches+1, head_dim)
            weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches+1, n_heads, head_dim)
            weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches+1, dim)

            return self.proj_drop(self.proj(weighted_avg))  # (n_samples, n_patches+1, dim)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, p: float = 0.):
        """
        :param in_features:
        :param hidden_features:
        :param out_features:
        :param p:
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p)
        )

    def forward(self, x):
        """
        :param x: torch.Tensor (n_samples, n_patches+1, in_features)
        :return: torch.Tensor (n_samples, n_patches+1, out_features)
        """
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, p: float = 0.,
                 attn_p: float = 0.1):
        """
        :param dim: Embedding dimension
        :param n_heads: Number of attention heads
        :param mlp_ratio: Determines the hidden dimension size of the MLP wrt dim
        :param qkv_bias: If true bias for qkv
        :param p: dropout rate
        :param attn_p: dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim, p=p)

    def forward(self, x):
        """
        :param x: torch.Tensor (n_samples, n_patches+1, dim)
        :return: torch.Tensor (n_samples, n_patches+1, dim)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size: int = 384,
            patch_size: int = 16,
            in_chans: int = 3,
            n_classes: int = 1_000,
            embed_dim: int = 768,
            depth: int = 12,
            n_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            p: float = 0.,
            attn_p: float = 0.
    ):
        """
        Parameters:
            :param img_size: Both height and width of the image (it is a square)
            :param patch_size: Both height and width of the image (it is a square)
            :param in_chans: Number of input channels
            :param n_classes:Number of classes
            :param embed_dim: Dimensionality of the token/patch embeddings
            :param depth: Number of transformer blocks
            :param n_heads: Number of attention heads
            :param mlp_ratio: Determines the hidden dimension size of the MLP wrt dim
            :param qkv_bias: If true bias for qkv
            :param p: dropout rate
            :param attn_p: dropout rate

        Attributes:
            patch_embed: PatchEmbed = Instance of `PatchEmbed` layer
            cls_token: nn.Parameter = Learnable parameter that will represent the first token in the sequence.
                                      It has `embed_dim` elements.
            pos_emb: nn.Parameter = Positional embedding of the cls token + all the patches.
                                    It has `(n_patches + 1) * embed_dim` elements.
            pos_drop: nn.Dropout = Dropout layer
            blocks: nn.ModuleList = List of `Block` modules
            norm: nn.LayerNorm = Layer Normalization
        """
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        ...