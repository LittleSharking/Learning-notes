
# 软件工程原理与实践实验报告

<center>姓名：李东旭  学号：23020007056</center>

| 姓名和学号         | 李东旭，23020007056                  |
| -------------------- | ------------------------------ |
| 课程 | 中国海洋大学25秋《软件工程原理与实践》 |
| 实验名称           | 实验5：ViT & Swin Transformer         |
## 一、实验内容
### 1. 视频学习
#### 1.1 Vision Transformer (ViT)
使用 pytorch 实现 ViT
```
import torch
import torch.nn as nn
from einops import rearrange

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu"
            ),
            num_layers=depth
        )
        self.to_cls = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, dim, H//patch, W//patch)
        B, C, H, W = x.shape
        x = rearrange(x, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=self.patch_size, p2=self.patch_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.to_cls(x[:, 0])
        return self.mlp_head(x)
```
#### 1.2 Swin Transformer
使用 pytorch 实现 Swin Transformer
```
import torch
import torch.nn as nn
import math

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, dropout=dropout)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.shift_size = window_size // 2

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, self.window_size * self.window_size, C)
        x = self.attn(x)
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24]):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            stage = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i),
                    input_resolution=(img_size // patch_size // (2 ** i), img_size // patch_size // (2 ** i)),
                    num_heads=heads[i],
                    window_size=7 if i == 0 else 14,
                    mlp_ratio=4.,
                    dropout=0.1
                ) for _ in range(depths[i])
            ])
            self.stages.append(stage)
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers-1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers-1)), num_classes)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, embed_dim, H//patch, W//patch)
        x = rearrange(x, "B (h p1) (w p2) c -> B (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return self.head(x)
```
### 2. 思考题
#### 2.1 在ViT中要降低 Attention的计算量，有哪些方法？（提示：Swin的 Window attention，PVT的attention）
答：ViT通过分块计算和动态筛选减少注意力计算量。通过将图像划分为局部窗口，仅在窗口内计算自注意；PVT通过逐层下采样特征图减少后续层的Token数量；MobileViT则用卷积生成重要性掩码，筛选高响应区域参与计算，减少冗余，核心是“局部优先+动态调整”的思路。


#### 2.2 Swin体现了一种什么思路？对后来工作有哪些启发？（提示：先局部再整体）
答：Swin的核心是融合CNN的层次化归纳偏置与Transformer的全局能力，通过窗口注意力降低计算量、移位窗口实现跨区域交互、层次化下采样构建多尺度特征，既保留局部感知又支持全局建模。这启发后续工作偏向混合架构、动态计算优化，以及硬件友好的规则化设计以适配边缘设备。


#### 2.3 有些网络将CNN和Transformer结合，为什么一般把 CNN block放在面前，Transformer block放在后面？
答：CNN前置主要为了高效下采样和保留局部特征。卷积能快速降低分辨率，减少Transformer的Token数量，缓解O(n²)计算压力。同时，CNN的局部感受野适合提取边缘、纹理等低级特征，为Transformer的全局注意力提供更优质的输入，避免直接处理高分辨率图像的高昂成本。


#### 2.4 阅读并了解Restormer，思考：Transformer的基本结构为 attention+ FFN，这个工作分别做了哪些改进？
答：Restormer针对Transformer的不足，提出滤波注意力、倒置残差连接、全局平均池化和轻量化MLP，在ImageNet上准确率达83.7%，推理速度比Swin快1.8倍，兼顾效率与性能。



## 二、问题总结与体会
通过本次ViT与Swin Transformer实现实验，我对二者设计逻辑与工程实践有了更直观认知。数据预处理中，ViT的patch embedding质量、Swin窗口大小与特征图分辨率的匹配直接影响特征提取效果；训练时，随机种子保障了结果可复现，Dropout虽添不确定性却有效防过拟合。不仅掌握了代码实现，更明白“模型适配任务”比“追新”更重要——未来做视觉任务，会先分析数据特性再选模型优势。
