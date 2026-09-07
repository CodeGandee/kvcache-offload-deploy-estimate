# Tracked external implementations

These repositories are pinned as Git submodules at the revisions inspected for the
current estimate. They provide source code and checkpoint metadata; the main project
does not vendor model weights.

| Path | Role | Pinned commit |
|---|---|---|
| `shadowkv` | ShadowKV algorithm and kernel reference | `e51904cdeab7d4d34013370f09f2cf5fcd655e15` |
| `infersim` | Analytical simulator and A800 cross-check | `d02a40c501ad7dca77873cb030395e6cbb8dc7ee` |
| `glm-5` | Official GLM serving/model code | `008de4dbcc220032eb9b80a9a9802afad46a4053` |
| `transformers-glm53` | Transformers revision containing GLM-5.3 support | `c93057d4835cd31752bb56f59989dd27696eb45b` |
| `kimi-k2.7-code` | Kimi checkpoint metadata and source/configuration | `74797c9c62378b951a1f6fcf5c4631024e9b8bef` |
| `deepseek-v4-flash` | V4 Flash checkpoint metadata and source/configuration | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| `llama-models` | Llama 3.1 reference model code | `0e0b8c519242d5833d8c11bffc1232b77ad7f301` |

Clone without large Git LFS payloads:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

Do not update a gitlink without documenting why the new revision changes—or does not
change—the assumptions and results.
