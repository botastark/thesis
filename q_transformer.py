import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core_qnn.quaternion_ops import *
from core_qnn.quaternion_layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def QNorm(x, eps):
    r, i, j, k = torch.chunk(x, chunks=4, dim=-1)
    qnorm = torch.sqrt(r * r + i * i + j * j + k * k + eps)
    r = r / qnorm
    i = i / qnorm
    j = j / qnorm
    k = k / qnorm

    return [r, i, j, k]


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model // 4
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        [r, i, j, k] = QNorm(x, self.eps)

        norm_r = self.alpha * r + self.bias
        norm_i = self.alpha * i + self.bias
        norm_j = self.alpha * j + self.bias
        norm_k = self.alpha * k + self.bias
        norm = torch.cat([norm_r, norm_i, norm_j, norm_k], dim=-1)

        return norm


def quarternion_multiplication(a, b, transpose=True):
    """Performs hamilton product between two quarternion sequences.
    a = (r,x,y,z)
    b = (r',x',y',z')
    following:
    (rr' - xx' - yy' - zz')  +
    (rx' + xr' + yz' - zy')i +
    (ry' - xz' + yr' + zx')j +
    (rz' + xy' - yx' + zr')k
    """

    ar, ax, ay, az = torch.chunk(a, chunks=4, dim=-1)
    br, bx, by, bz = torch.chunk(b, chunks=4, dim=-1)

    if transpose == True:
        if len(br.shape) > 2:
            r = (
                torch.matmul(ar, br.transpose(-2, -1))
                - torch.matmul(ax, bx.transpose(-2, -1))
                - torch.matmul(ay, by.transpose(-2, -1))
                - torch.matmul(az, bz.transpose(-2, -1))
            )
            i = (
                torch.matmul(ar, bx.transpose(-2, -1))
                + torch.matmul(ax, br.transpose(-2, -1))
                + torch.matmul(ay, bz.transpose(-2, -1))
                - torch.matmul(az, by.transpose(-2, -1))
            )
            j = (
                torch.matmul(ar, by.transpose(-2, -1))
                - torch.matmul(ax, bz.transpose(-2, -1))
                + torch.matmul(ay, br.transpose(-2, -1))
                + torch.matmul(az, bx.transpose(-2, -1))
            )
            k = (
                torch.matmul(ar, bz.transpose(-2, -1))
                + torch.matmul(ax, by.transpose(-2, -1))
                - torch.matmul(ay, bx.transpose(-2, -1))
                + torch.matmul(az, br.transpose(-2, -1))
            )

        else:
            r = (
                torch.matmul(ar, br.t())
                - torch.matmul(ax, bx.t())
                - torch.matmul(ay, by.t())
                - torch.matmul(az, bz.t())
            )
            i = (
                torch.matmul(ar, bx.t())
                + torch.matmul(ax, br.t())
                + torch.matmul(ay, bz.t())
                - torch.matmul(az, by.t())
            )
            j = (
                torch.matmul(ar, by.t())
                - torch.matmul(ax, bz.t())
                + torch.matmul(ay, br.t())
                + torch.matmul(az, bx.t())
            )
            k = (
                torch.matmul(ar, bz.t())
                + torch.matmul(ax, by.t())
                - torch.matmul(ay, bx.t())
                + torch.matmul(az, br.t())
            )
    else:
        r = (
            torch.matmul(ar, br)
            - torch.matmul(ax, bx)
            - torch.matmul(ay, by)
            - torch.matmul(az, bz)
        )
        i = (
            torch.matmul(ar, bx)
            + torch.matmul(ax, br)
            + torch.matmul(ay, bz)
            - torch.matmul(az, by)
        )
        j = (
            torch.matmul(ar, by)
            - torch.matmul(ax, bz)
            + torch.matmul(ay, br)
            + torch.matmul(az, bx)
        )
        k = (
            torch.matmul(ar, bz)
            + torch.matmul(ax, by)
            - torch.matmul(ay, bx)
            + torch.matmul(az, br)
        )

    return torch.cat([r, i, j, k], dim=-1)


def ComponentActivation(q, act_func=F.gelu):
    scores_r, scores_i, scores_j, scores_k = torch.chunk(q, 4, dim=-1)
    if act_func == F.softmax:
        scores_r = act_func(scores_r, dim=-1)
        scores_i = act_func(scores_i, dim=-1)
        scores_j = act_func(scores_j, dim=-1)
        scores_k = act_func(scores_k, dim=-1)
    else:
        scores_r = act_func(scores_r)
        scores_i = act_func(scores_i)
        scores_j = act_func(scores_j)
        scores_k = act_func(scores_k)

    scores = torch.cat([scores_r, scores_i, scores_j, scores_k], dim=-1)
    return scores


# TODO not sure about scale applied to q only
class Attention(nn.Module):
    def __init__(
        self,
        dim,  # embed_size
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = QuaternionLinearAutograd(dim, dim * 3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = Norm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = QuaternionLinearAutograd(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale

        attn = quarternion_multiplication(q, k)
        # print("Att shape", attn.shape)

        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)

        attn = ComponentActivation(attn, act_func=F.softmax)
        attn = self.attn_drop(attn)

        x = quarternion_multiplication(attn, v, transpose=False)

        # print("x shape att v", x.shape)
        x = x.transpose(1, 2).reshape(B, N, C)
        # print("x shape after transpose and reshape", x.shape)

        x = self.proj(x)
        # print("x shape after lin", x.shape)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # drop_probs = to_2tuple(drop)

        self.fc1 = QuaternionLinearAutograd(in_features, hidden_features)
        # self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = QuaternionLinearAutograd(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = ComponentActivation(x, act_func=F.gelu)
        # x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=Norm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            # act_layer=act_layer,
            drop=drop,
        )
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
