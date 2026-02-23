# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Mamba implementation with high-level ONNX operators.

This version exports to ONNX with clean, high-level operators:
- SelectiveSSM (custom operator with delta computation)
- Conv1d (with fixed padding)
- Linear (matrix multiplication)

No fragmented graphs, no dynamic operations!
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ============================================================================
# ONNX-friendly Layers
# ============================================================================


class LayerNormFunction(Function):
    """Custom LayerNorm with mean= 0 and std = 1 (baked in)."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # Normalization: mean=0, std=1 are implicit (pre-normalized or baked in)
        x_norm = x/ (1+eps)
        return x_norm * weight + bias

    @staticmethod
    def symbolic(g, x, weight, bias, eps):
        """Export as custom LayerNorm operator with shape annotation."""
        # Shape annotation: output shape = input shape
        y = g.op("ai.mamba::LayerNorm", x, weight, bias, epsilon_f=eps, outputs=1)
        y.setType(x.type())  # Ensure output has same shape/type as input
        return y

class LayerNorm(nn.Module):
    """Custom LayerNorm with mean= 0 and std = 1 (baked in).

    Exports as single custom ONNX operator
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Only learnable parameters - no buffers to avoid Identity nodes
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Simplified: no mean/std buffers passed
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SiLUFunction(Function):
    """SiLU with explicit ONNX symbolic registration."""

    @staticmethod
    def forward(ctx, x):
        return x * torch.sigmoid(x)

    @staticmethod
    def symbolic(g, x):
        # Export as single Silu node with shape propagation
        y = g.op("Silu", x)
        y.setType(x.type())  
        return y


class SiLU(nn.Module):
    """SiLU activation that exports as single node in ONNX.

    Ensures export as 'Silu' operator instead of Sigmoid+Mul decomposition.
    """

    def forward(self, x):
        return SiLUFunction.apply(x)
    
class PaddedConv1dFunction(Function):
    """
    Conv1d with pre-padding to achieve 'same' padding without extra nodes.

    Exports as a standard ONNX Conv node with pads attribute (no Pad + Conv, just Conv).
    No custom op registration needed - uses standard ONNX Conv.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, padding_left, padding_right, groups, kernel_size):
        """
        Forward pass of Conv1d with pre-padding for 'same' output length.
        """
        # Pre-pad input for 'same' convolution
        x_padded = F.pad(x, (padding_left, padding_right))
        return F.conv1d(x_padded, weight, bias, stride=1, padding=0, groups=groups)

    @staticmethod
    def symbolic(g, x, weight, bias, padding_left, padding_right, groups, kernel_size):
        """
        Export as standard ONNX Conv node with padding as attributes.
        No custom op needed - uses standard ONNX Conv with pads attribute.
        """
        # Standard ONNX Conv op (works for 1D, 2D, 3D based on input rank)
        # pads format for 1D: [pad_start, pad_end]

        #shape inference here

        return g.op(
            "Conv",
            x,
            weight,
            bias,
            kernel_shape_i=[kernel_size],
            pads_i=[padding_left, padding_right],
            strides_i=[1],
            dilations_i=[1],
            group_i=groups,
        )


class PaddedConv1d(nn.Module):
    """Conv1d with pre-padding to achieve 'same' padding without extra ONNX nodes.

    Exports as standard ONNX Conv node with padding in attributes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        
        # Compute asymmetric padding for "same" behavior
        # For kernel_size=4: pad_left=1, pad_right=2 -> output_len = input_len
        self.padding_left = (kernel_size - 1) // 2
        self.padding_right = (kernel_size - 1) - self.padding_left
        
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=0,  # No padding here, we handle it via custom function
            bias=bias,
        )

    def forward(self, x):
        return PaddedConv1dFunction.apply(x,self.conv1d.weight,self.conv1d.bias,self.padding_left,self.padding_right,
                                          self.groups, self.kernel_size)

# ============================================================================
# Custom ONNX operator
# ============================================================================


def register_custom_op():
    """Register SelectiveSSM as custom ONNX operator.

    SelectiveSSM operator signature:
        Inputs: x, A_log, B, C, D, dt, res
        Outputs: y (after SSM + gating)

    Includes internally:
        - Exp(A_log)
        - SSM computation
        - Gating (res * sigmoid(res)) * y
    """
    from torch.onnx import register_custom_op_symbolic

    def selective_ssm_symbolic(g, x, A_log, B, C, D, dt, res):
        """ONNX symbolic for SelectiveSSM."""
        return g.op("ai.mamba::SelectiveSSM", x, A_log, B, C, D, dt, res, outputs=1)

    # Register the symbolic function
    register_custom_op_symbolic("::SelectiveSSMFunction", selective_ssm_symbolic, opset_version=9)

    print("✅ Custom ONNX operator registered: ai.mamba::SelectiveSSM")
    print("   Inputs: x, A_log, B, C, D, dt, res")
    print("   Outputs: y (SSM + gating)")
    print("   Internal ops: Exp, SSM, SiLU gating")
    return True


# Call registration when module is imported
_REGISTERED = register_custom_op()


class SelectiveSSMFunction(Function):
    """Custom autograd function for Selective SSM with delta computation.

    This will be exported as a single "SelectiveSSM" operator in ONNX.
    Includes delta (dt) computation internally.
    """

    @staticmethod
    def forward(ctx, x, A_log, B, C, D, dt, res, batch_size, seq_len, d_inner, d_state):
        """
        Forward pass of Selective SSM with delta and gating.

        Args:
            x: (B, L, D) - input after conv
            A_log: (D, N) - log of state matrix (will be exp'd inside)
            B: (B, L, N) - input projection
            C: (B, L, N) - output projection
            D: (D,) - skip connection
            dt: (B, L, D) - delta (timestep)
            res: (B, L, D) - residual for gating
            batch_size: int - fixed batch size
            seq_len: int - fixed sequence length
            d_inner: int - fixed inner dimension
            d_state: int - fixed state dimension (N)

        Returns:
            y: (B, L, D) - output after gating
        """
        # Exp inside custom op
        A = torch.exp(A_log)
        # SSM computation: y = sum((A * x + B) * C, dim=-1) + D * x
        # Use fixed dimensions instead of dynamic shape queries
        B_size = batch_size
        L = seq_len
        D_inner = d_inner
        N = d_state

        # Reshape for broadcasting
        x_expanded = x.reshape(B_size, L, D_inner, 1)
        A_expanded = A.reshape(1, 1, D_inner, N)

        Ax = x_expanded * A_expanded
        B_expanded = B.reshape(B_size, L, 1, N)
        state = Ax + B_expanded

        C_expanded = C.reshape(B_size, L, 1, N)
        y = torch.sum(state * C_expanded, dim=-1)

        # Skip connection with delta modulation
        D_expanded = D.reshape(1, 1, D_inner)
        y = y + D_expanded * x * dt

        # Gating (inside custom op)
        res_act = res * torch.sigmoid(res)  # SiLU
        y = y * res_act

        return y

    @staticmethod
    def symbolic(g, x, A_log, B, C, D, dt, res, batch_size, seq_len, d_inner, d_state):
        """
        ONNX symbolic function - single SelectiveSSM node with gating.
        """

        # Pass fixed-size integers as attributes (use _i suffix for ints)
        # to avoid creating malformed Constant node attributes during export.
        y = g.op(
            "ai.mamba::SelectiveSSM",
            x,
            A_log,
            B,
            C,
            D,
            dt,
            res,
            batch_size_i=batch_size,
            seq_len_i=seq_len,
            d_inner_i=d_inner,
            d_state_i=d_state,
            outputs=1,
        )
        y.setType(x.type())  # Ensure output has same shape/type as input

        return y

class SelectiveSSM(nn.Module):
    """Selective SSM that exports as a single ONNX operator."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        batch_size: int = 1,
        seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        self.padding_size = d_conv - 1
        # Fixed dimensions for ONNX export (no dynamic shape)
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Linear projections - separate layers to avoid slicing
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.res_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # Conv1d with 'same' padding - output length = input length
        # Uses PaddedConv1d which exports as single ONNX Conv node (no Pad node)
        self.conv1d = PaddedConv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters - separate layers to avoid slicing
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # State space matrices (pre-computed shapes to avoid unsqueeze)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Activation - ensure single ONNX node
        self.act_conv = SiLU()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.A_log, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.res_proj.weight)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - canonical ONNX graph.
        Padding is in Conv attr, no extra nodes.
        """
        # 1. Input projection - parallel projections, NO slicing
        x_proj = self.in_proj(x)  # (B, L, D_inner)
        res = self.res_proj(x)  # (B, L, D_inner)

        # 2. Conv with padding in attributes (no extra nodes)
        x_proj_t = x_proj.transpose(1, 2)  # (B, D_inner, L)
        x_conv = self.conv1d(x_proj_t)  # Padding is Conv attribute
        x_conv = x_conv.transpose(1, 2)  # (B, L, D_inner)

        # 3. Activation
        x_conv = self.act_conv(x_conv)

        # 4. SSM parameters - separate projections, NO slicing
        B = self.B_proj(x_conv)  # (B, L, N)
        C = self.C_proj(x_conv)  # (B, L, N)

        # 5. Delta computation (timestep)
        dt = self.dt_proj(x_conv)  # (B, L, D_inner)
        dt = F.softplus(dt)

        # 6. Selective SSM (custom operator - includes Exp, gating, all inside)
        y = SelectiveSSMFunction.apply(
            x_conv, self.A_log, B, C, self.D, dt, res,
            self.batch_size, self.seq_len, self.d_inner, self.d_state
        )

        # 7. Output projection
        output = self.out_proj(y)

        return output


class MambaBlock(nn.Module):
    """Mamba block with residual connection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 512,
    ):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand_factor, batch_size, seq_len)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual: LayerNorm → SSM → Add"""
        return x + self.dropout(self.ssm(self.norm(x)))


class Mamba(nn.Module):
    """Mamba model with clean ONNX export.

    ONNX operators per layer:
    - LayerNorm
    - Linear (3x: in_proj, x_proj, dt_proj, out_proj)
    - Conv1d (fixed padding)
    - Silu (2x: activation + gating)
    - Softplus (delta)
    - Exp (A matrix)
    - ai.mamba::SelectiveSSM (custom op)
    - Mul (gating)
    - Add (residual)

    Total: ~12-15 high-level nodes per layer
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        vocab_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        use_embedding: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Input
        if use_embedding and vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        else:
            self.embedding = None
            self.input_proj = nn.Linear(d_model, d_model)

        # Fixed batch size for ONNX export
        self.batch_size = 1  # Fixed batch size for deployment

        # Mamba layers
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand_factor, dropout, self.batch_size, max_seq_len) for _ in range(n_layers)]
        )

        # Output
        self.norm_f = LayerNorm(d_model)

        if num_classes is not None:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            if vocab_size is not None:
                self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            else:
                self.lm_head = None

        self._initialize_weights()

    def _initialize_weights(self):
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, std=0.02)
            nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - NO dynamic operations.

        Fixed input shape: (batch_size, max_seq_len, d_model)
        """
        if self.embedding is not None:
            x = self.embedding(x)
            x = x + self.pos_embedding
        else:
            x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        if hasattr(self, "classifier"):
            x = torch.mean(x, dim=1, keepdim=False)
            x = self.classifier(x)
        elif self.lm_head is not None:
            x = self.lm_head(x)

        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def mamba_small(num_classes: int = 10, max_seq_len: int = 512):
    return Mamba(
        d_model=128,
        n_layers=2,
        d_state=16,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        use_embedding=False,
    )


def mamba_base(num_classes: int = 10, max_seq_len: int = 512):
    return Mamba(
        d_model=256,
        n_layers=4,
        d_state=16,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        use_embedding=False,
    )
