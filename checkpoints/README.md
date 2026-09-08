# Checkpoints

Model weights are intentionally excluded from Git. The stable local layout is:

```text
checkpoints/
├── pretrained/
│   ├── Qwen3-VL-8B-Instruct/
│   ├── Qwen3-VL-4B-Instruct/
│   └── Qwen3-Embedding-0.6B/
├── tools/
│   ├── grounding-dino-base/
│   └── DeepSeek-OCR-updated/
└── released/
    └── EventMemAgent-8B/
```

Qwen3-VL-8B-Instruct is the GRPO-trained EventMemAgent backbone. Qwen3-VL-4B-Instruct is the frozen model used to generate long-term-memory event captions during preprocessing. You may use local paths or compatible Hugging Face/ModelScope model identifiers where the underlying loader supports them.

The [EventMemAgent-8B checkpoint](https://huggingface.co/lingcco/EventMemAgent-8B)
is available. Download on your compute server:

```bash
hf download lingcco/EventMemAgent-8B --local-dir checkpoints/released/EventMemAgent-8B
```

The release contains merged model weights, configuration, tokenizer and processor
files. Optimizer states, intermediate checkpoints and training logs are not included.

Do not commit access tokens, private hub URLs, optimizer states containing sensitive metadata, or unreviewed experiment logs.

The inspected experiment launcher uses the compatibility-updated OCR artifact
`prithivMLmods/DeepSeek-OCR-Latest-BF16.I64` (its card identifies
`deepseek-ai/DeepSeek-OCR` as the base model), stored as `DeepSeek-OCR-updated`.
The original DeepSeek-OCR remote code fails to import `LlamaFlashAttention2`
under the policy environment's Transformers 4.57.3. Do not silently swap these
artifacts or downgrade the policy environment. Pin the actual OCR revision
alongside the released checkpoint after the release audit.
