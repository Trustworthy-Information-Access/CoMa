import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.w12 = nn.Linear(input_dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class GatedPoolingSwiGLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None):
        super().__init__()
        self.gate_layer = nn.Linear(input_dim, 1)
        self.temperature = 2.0
        hidden_dim = hidden_dim or input_dim * 2

        self.mlp = SwiGLUFFN(input_dim, hidden_dim, input_dim)  # 先映射回 input_dim 保持维度

        self.out_proj = nn.Linear(input_dim, output_dim)     # pooling 后再映射到输出维度

    def forward(self, x):
        """
        x: (B, N, D)
        return: (B, output_dim)
        """
        x_transformed = self.mlp(x)  # (B, N, D)
        gate_logits = self.gate_layer(x_transformed).squeeze(-1)  # (B, N)
        gate_weights = F.softmax(gate_logits / self.temperature, dim=-1)  # (B, N)
        pooled = torch.sum(x_transformed * gate_weights.unsqueeze(-1), dim=1)  # (B, D)
        embedding = self.out_proj(pooled)  # (B, output_dim)

        return embedding


class LatentAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable query: [L, D]
        self.latent_query = nn.Parameter(torch.randn(num_latents, input_dim))

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, D)
        Returns:
            embedding: Output tensor of shape (B, output_dim)
        """
        B, N, D = x.shape

        # Expand latent query for batch
        q = self.latent_query.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        
        # Reshape to multi-heads
        q = q.view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)                  # (B, H, N, d)
        v = x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)                  # (B, H, N, d)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale                 # (B, H, L, N)
        attn_weights = F.softmax(attn_scores, dim=-1)                                   # (B, H, L, N)
        attended = torch.matmul(attn_weights, v)                                        # (B, H, L, d)

        # Merge heads
        attended = attended.transpose(1, 2).contiguous().view(B, self.num_latents, D)   # (B, L, D)

        # MLP over latent tokens
        attended = self.mlp(attended)                                                   # (B, L, output_dim)
        # Mean pooling over latent dimension
        embedding = attended.mean(dim=1)                                                # (B, output_dim)

        return embedding
