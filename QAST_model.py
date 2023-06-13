import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from core_qnn.quaternion_ops import *
from core_qnn.quaternion_layers import *
from timm.models.layers import to_2tuple, trunc_normal_
from q_transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Patch using convensional CNN as in original paper
# override the timm package to relax the input shape constraint.


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=256, patch_size=16, stride=10, in_chans=4, embed_dim=768
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        # self.num_patches = num_patches

        # TODO compare qconv or conv works for patchembedding
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.projq = QuaternionConv(in_chans, embed_dim, patch_size, stride)

    def forward(self, x):
        zeros = torch.zeros(x.shape).to(device)
        x = torch.cat((zeros, x, x, x), 1)
        # print("Qx shape: [0, g, g, g]", x.shape)
        x = self.projq(x).flatten(2).transpose(1, 2)
        return x


class QASTModel(nn.Module):
    def __init__(
        self,
        label_dim=527,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        audioset_pretrain=False,
        model_size="base384",
        verbose=True,
        best_model=None,
    ):
        super(QASTModel, self).__init__()
        # automatcially get the intermediate shape
        self.original_embedding_dim = 768
        num_heads = 12
        mlp_ratio = 4.0
        qkv_bias = True
        qk_norm = False
        drop_rate = 0.0
        attn_drop_rate = 0.0
        depth = 12

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim

        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.original_embedding_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.original_embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 2, self.original_embedding_dim)
        )
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=0.0)

        # TODO pretrained or sinusoidal
        if imagenet_pretrain and best_model != None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # args_usepretrain.loss_fn = nn.BCEWithLogitsLoss()
            sd = torch.load(best_model, map_location=device)
            n_class = 10  # dep on pretraining dataset for CIFAR10-10, ImageNEt-1000, tinyImageNet-200
            new_audio_model = QASTModel(
                label_dim=n_class, input_tdim=128, imagenet_pretrain=False
            )

            audio_model = torch.nn.DataParallel(new_audio_model)
            audio_model.load_state_dict(sd)

            self.blocks = audio_model.blocks
        else:
            self.blocks = nn.Sequential(
                *[
                    Block(
                        self.original_embedding_dim,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                    )
                    for i in range(depth)
                ]
            )
        self.norm = Norm(self.original_embedding_dim)

        # Classifier Head
        self.fc_norm = Norm(self.original_embedding_dim)
        self.head = (
            nn.Linear(self.original_embedding_dim, label_dim)
            if label_dim > 0
            else nn.Identity()
        )

    @autocast()
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        x = (x[:, 0] + x[:, 1]) / 2
        x = self.fc_norm(x)
        x = self.head(x)
        return x

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
