# vllm-gravity-plugin

vLLM plugin for [Trillion Labs](https://trillionlabs.co) Gravity-MoE models.

Registers `GravityMoEForCausalLM` as a supported architecture in vLLM, enabling direct inference and serving of Gravity-MoE models without any code changes to vLLM.

## Installation

```bash
pip install -e .
```

## Usage

Once installed, vLLM automatically discovers the plugin. No additional configuration needed.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="trillionlabs/Gravity-16B-A3B-Base",
    trust_remote_code=True,
    dtype="bfloat16",
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Explain quantum computing."], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Serving

```bash
vllm serve trillionlabs/Gravity-16B-A3B-Base \
    --trust-remote-code \
    --dtype bfloat16
```

## How It Works

Gravity-MoE shares the same underlying architecture as DeepSeek V3 (MLA attention + MoE with shared experts). This plugin registers `GravityMoEForCausalLM` to use vLLM's existing optimized `DeepseekV3ForCausalLM` implementation, giving you full access to vLLM's performance optimizations (PagedAttention, tensor parallelism, continuous batching, etc.) with zero overhead.

## Supported Models

| Model | HuggingFace |
|---|---|
| Gravity-16B-A3B-Base | [trillionlabs/Gravity-16B-A3B-Base](https://huggingface.co/trillionlabs/Gravity-16B-A3B-Base) |
| Gravity-16B-A3B-Preview | [trillionlabs/Gravity-16B-A3B-Preview](https://huggingface.co/trillionlabs/Gravity-16B-A3B-Preview) |

## License

Apache 2.0
