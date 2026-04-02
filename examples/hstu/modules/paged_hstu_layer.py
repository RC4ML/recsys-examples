# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import cast
from configs import InferenceHSTUConfig, KVCacheConfig
from modules.ffn_layer import FFNLayer
from modules.jagged_data import JaggedData
from modules.paged_hstu_infer_layer import PagedHSTUInferLayer


class PagedHSTULayer(torch.nn.Module):
    """
    Composite inference layer:
    one attention layer (PagedHSTUInferLayer) followed by one FFNLayer.
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
        kvcache_config: KVCacheConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._attn_layer = PagedHSTUInferLayer(config, kvcache_config, layer_idx)
        self._ffn_layer = FFNLayer(config)
        # Keep output buffer contract used by HSTUBlockInference cudagraph path.
        self.output_buffer_ = self._ffn_layer.output_buffer_

    @torch.inference_mode()
    def forward(self, jd: JaggedData) -> JaggedData:
        batch_size = jd.seqlen.shape[0]
        num_tokens = jd.values.shape[0]
        hidden_states = self.forward_naive(
            batch_size=batch_size,
            num_tokens=num_tokens,
            layer_input=jd.values,
            jd=jd,
            kv_cache_metadata=None,
        )
        jd.values = hidden_states
        return jd

    @torch.inference_mode()
    def forward_naive(
        self,
        batch_size: int,
        num_tokens: int,
        layer_input: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> torch.Tensor:
        hidden_states = cast(
            torch.Tensor,
            self._attn_layer.forward_naive(
                batch_size,
                num_tokens,
                layer_input,
                jd,
                kv_cache_metadata,
            ),
        )
        return self._ffn_layer.forward_naive(
            num_tokens,
            hidden_states,
        )

    @torch.inference_mode()
    def forward_input(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._attn_layer.forward_input(
                batch_size,
                num_tokens,
                input_buffer,
                jd,
                kv_cache_metadata,
            ),
        )

    @torch.inference_mode()
    def forward_output(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> torch.Tensor:
        attn_output = cast(
            torch.Tensor,
            self._attn_layer.forward_output(
                batch_size,
                num_tokens,
                input_buffer,
                jd,
                kv_cache_metadata,
            ),
        )
        return self._ffn_layer.forward_naive(
            num_tokens,
            attn_output,
        )