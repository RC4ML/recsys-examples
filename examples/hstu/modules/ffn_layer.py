# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from configs import InferenceHSTUConfig
from ops.pt_ops.torch_addmm import torch_addmm_silu_fwd
from ops.triton_ops.triton_addmm import triton_addmm_silu_fwd


class FFNLayer(torch.nn.Module):
    """
    Transformer FFN layer for inference.

    x = ln(x)
    h = silu(up_proj(x))
    y = down_proj(h) + residual
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
    ):
        super().__init__()
        self._embedding_dim: int = config.hidden_size
        self._ffn_hidden_dim: int = getattr(config, "ffn_hidden_size", 4 * config.hidden_size)
        self._eps = config.layernorm_epsilon
        self._residual = config.residual

        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        device = torch.cuda.current_device()

        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None

        self._linear_up = torch.nn.Linear(
            self._embedding_dim,
            self._ffn_hidden_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self._linear_down = torch.nn.Linear(
            self._ffn_hidden_dim,
            self._embedding_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        for param in self._linear_up.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))
        for param in self._linear_down.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))

        self._linear_up_weight = self._linear_up.weight.T.contiguous()
        self._linear_down_weight = self._linear_down.weight.T.contiguous()

        sm = torch.cuda.get_device_properties(0).major
        if sm == 8:
            self.addmm_silu_impl = triton_addmm_silu_fwd
        elif sm == 9:
            self.addmm_silu_impl = torch_addmm_silu_fwd
        else:
            raise ValueError(f"Unsupported SM major version: {sm}")

    def up_addmm_silu_impl(self, input_data, num_tokens):
        if num_tokens >= 2048:
            _, silu_output_data = self.addmm_silu_impl(
                x=input_data,
                w=self._linear_up_weight,
                y=self._linear_up.bias,
                silu=True,
            )
        else:
            silu_output_data = self._linear_up(input_data)
            F.silu(silu_output_data, inplace=True)
        return silu_output_data

    def down_addmm_impl(self, input_data, residual, num_tokens):
        if num_tokens >= 2048:
            output_data, _ = self.addmm_silu_impl(
                x=input_data,
                w=self._linear_down_weight,
                y=residual,
                silu=False,
            )
        else:
            output_data = self._linear_down(input_data)
            torch.add(output_data, residual, out=output_data)
        return output_data

    @torch.inference_mode()
    def forward_naive(
        self,
        num_tokens: int,
        layer_input: torch.Tensor,
    ) -> torch.Tensor:
        # print(f"FFNLayer forward_naive with num_tokens={num_tokens}, layer_input.shape={layer_input.shape}")
        normed_input = F.layer_norm(
            layer_input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )
        hidden = self.up_addmm_silu_impl(normed_input, num_tokens)
        if self._residual:
            output = self.down_addmm_impl(hidden, layer_input, num_tokens)
        else:
            output = self._linear_down(hidden)
        return output