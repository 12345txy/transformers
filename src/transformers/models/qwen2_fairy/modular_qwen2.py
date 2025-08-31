from typing import Callable, Optional
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ...utils.import_utils import get_torch_version
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mistral.modeling_mistral import MistralModel
from .configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features % 2 != 0 or out_features % 2 != 0:
            raise ValueError("in_features and out_features must be even for ComplexLinear.")

        self.in_features = in_features
        self.out_features = out_features
        self.m = in_features // 2
        self.n = out_features // 2

        self.U_re = nn.Parameter(torch.empty(self.n, self.m))
        self.U_im = nn.Parameter(torch.empty(self.n, self.m))
        self.W_re = nn.Parameter(torch.empty(self.n, self.m))
        self.W_im = nn.Parameter(torch.empty(self.n, self.m))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicating a similar initialization strategy to nn.Linear's kaiming_uniform_
        # The reconstructed real matrix A' is composed of sums/diffs of U and W components.
        # To keep the variance of A's elements in a similar range as a standard Linear layer,
        # we initialize U and W with a smaller variance.
        stdv = 1.0 / math.sqrt(self.m) / 2.0
        self.U_re.data.uniform_(-stdv, stdv)
        self.U_im.data.uniform_(-stdv, stdv)
        self.W_re.data.uniform_(-stdv, stdv)
        self.W_im.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # Bias initialization similar to nn.Linear
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(self.out_features, self.in_features))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct the equivalent real-valued weight matrix A'
        # A' = [[U_re + W_re, W_im - U_im],
        #       [U_im + W_im, U_re - W_re]]
        A_11 = self.U_re + self.W_re
        A_12 = self.W_im - self.U_im
        A_21 = self.U_im + self.W_im
        A_22 = self.U_re - self.W_re

        # Assemble A' from blocks
        A_prime_top = torch.cat([A_11, A_12], dim=1)
        A_prime_bottom = torch.cat([A_21, A_22], dim=1)
        A_prime = torch.cat([A_prime_top, A_prime_bottom], dim=0)

        return F.linear(x, A_prime, self.bias)

    def from_linear(self, linear_layer: nn.Linear):
        """
        Initializes the ComplexLinear layer's weights from a standard nn.Linear layer.
        """
        if linear_layer.in_features != self.in_features or linear_layer.out_features != self.out_features:
            raise ValueError("Linear layer dimensions do not match ComplexLinear dimensions.")

        A_prime = linear_layer.weight.data
        # Split the weight matrix into four blocks
        A_11 = A_prime[: self.n, : self.m]
        A_12 = A_prime[: self.n, self.m :]
        A_21 = A_prime[self.n :, : self.m]
        A_22 = A_prime[self.n :, self.m :]

        # Decompose the blocks into U and W components based on the math principle
        self.U_re.data = 0.5 * (A_11 + A_22)
        self.U_im.data = 0.5 * (A_21 - A_12)
        self.W_re.data = 0.5 * (A_11 - A_22)
        self.W_im.data = 0.5 * (A_12 + A_21)

        if linear_layer.bias is not None:
            if self.bias is not None:
                self.bias.data.copy_(linear_layer.bias.data)
        elif self.bias is not None:
            self.bias.data.zero_()


class Qwen2MLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = ComplexLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = ComplexLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = ComplexLinear(self.intermediate_size, self.hidden_size, bias=False)


class Qwen2Attention(LlamaAttention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = ComplexLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = ComplexLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = ComplexLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = ComplexLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


if version.parse(get_torch_version()) >= version.parse("2.3.0"):

    class Qwen2RMSNorm(nn.RMSNorm):
        def __init__(self, hidden_size, eps: float = 1e-6) -> None:
            super().__init__(normalized_shape=hidden_size, eps=eps, elementwise_affine=True)

else:

    @use_kernel_forward_from_hub("RMSNorm")
    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps: float = 1e-6) -> None:
            """
            Qwen2RMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

        def extra_repr(self):
            return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]


class Qwen2PreTrainedModel(LlamaPreTrainedModel):
    pass


class Qwen2Model(MistralModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen2ForCausalLM(LlamaForCausalLM):
    pass


class Qwen2ForSequenceClassification(LlamaForSequenceClassification):
    pass


class Qwen2ForTokenClassification(LlamaForTokenClassification):
    pass


class Qwen2ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "Qwen2PreTrainedModel",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "Qwen2RMSNorm",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2ForQuestionAnswering",
]

if __name__ == "__main__":
    def test_complex_linear_equivalence():
        print("Running equivalence test for ComplexLinear module...")
        in_features = 128
        out_features = 256
        batch_size = 10

        # Test with bias
        linear_bias = nn.Linear(in_features, out_features, bias=True)
        complex_linear_bias = ComplexLinear(in_features, out_features, bias=True)
        complex_linear_bias.from_linear(linear_bias)
        input_tensor = torch.randn(batch_size, in_features)
        output_linear = linear_bias(input_tensor)
        output_complex = complex_linear_bias(input_tensor)
        
        assert torch.allclose(output_linear, output_complex, atol=1e-6), "Equivalence test failed with bias!"
        print("Equivalence test with bias PASSED.")

        # Test without bias
        linear_no_bias = nn.Linear(in_features, out_features, bias=False)
        complex_linear_no_bias = ComplexLinear(in_features, out_features, bias=False)
        complex_linear_no_bias.from_linear(linear_no_bias)
        output_linear = linear_no_bias(input_tensor)
        output_complex = complex_linear_no_bias(input_tensor)

        assert torch.allclose(output_linear, output_complex, atol=1e-6), "Equivalence test failed without bias!"
        print("Equivalence test without bias PASSED.")

    test_complex_linear_equivalence()
