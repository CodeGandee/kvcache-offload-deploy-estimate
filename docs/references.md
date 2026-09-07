# References and pinned implementations

## Primary material

- Sun et al., “ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM
  Inference,” 2025.
- [ShadowKV implementation](https://github.com/ByteDance-Seed/ShadowKV)
- [InferSim](https://github.com/alibaba/InferSim)
- [GenZ-LLM-Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer)
- [LLMServingSim](https://github.com/casys-kaist/LLMServingSim)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Lenovo NVIDIA A800 product guide](https://lenovopress.lenovo.com/lp1813.pdf)

## Model code and checkpoint metadata

- [Kimi K2.7 Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code)
- [GLM-5](https://github.com/zai-org/GLM-5)
- [GLM-5.3 checkpoint](https://huggingface.co/zai-org/GLM-5.3)
- [GLM-5.3-Flash checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [vLLM GLM-5 recipe](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.yaml)
- [vLLM DeepSeek V4 Flash recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)
- [Meta Llama models](https://github.com/meta-llama/llama-models)

Exact revisions used for code inspection are pinned as submodules in
`extern/tracked/`. DeepSeek V4 Pro is deliberately excluded from the published case.
