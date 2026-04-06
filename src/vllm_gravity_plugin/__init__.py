"""vLLM plugin for Trillion Labs Gravity-MoE models."""

import logging

logger = logging.getLogger(__name__)
_registered = False


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

    _registered = True
