#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
uv pip install --system fire einops datasets tiktoken wandb jaxtyping tensorboard
