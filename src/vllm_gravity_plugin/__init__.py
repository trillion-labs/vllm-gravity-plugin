"""vLLM plugin for Trillion Labs Gravity-MoE models."""

import logging

logger = logging.getLogger(__name__)
_registered = False


def _register_mla_model_type() -> None:
    """Patch vLLM's MLA detection to recognize gravity_moe as an MLA model type."""
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    _original_is_deepseek_mla = ModelArchConfigConvertorBase.is_deepseek_mla

    def _patched_is_deepseek_mla(self) -> bool:
        model_type = getattr(self.hf_text_config, "model_type", None)
        if model_type == "gravity_moe":
            return (
                getattr(self.hf_text_config, "kv_lora_rank", None) is not None
            )
        return _original_is_deepseek_mla(self)

    ModelArchConfigConvertorBase.is_deepseek_mla = _patched_is_deepseek_mla


def register() -> None:
    """Register Gravity-MoE model architectures with vLLM.

    Maps GravityMoEForCausalLM to vLLM's DeepseekV3ForCausalLM implementation,
    which shares the same underlying architecture (MLA + MoE with shared experts).

    This function is re-entrant and safe to call multiple times.
    """
    global _registered

    if _registered:
        return

    from vllm import ModelRegistry

    supported = ModelRegistry.get_supported_archs()

    if "GravityMoEForCausalLM" not in supported:
        ModelRegistry.register_model(
            "GravityMoEForCausalLM",
            "vllm.model_executor.models.deepseek_v2:DeepseekV3ForCausalLM",
        )
        logger.info("Registered GravityMoEForCausalLM with vLLM")

    _register_mla_model_type()

    _registered = True
