# Acme-based reinforcement learning

Reinforcement learning based on [DeepMind's Acme](https://github.com/google-deepmind/acme/tree/master/) framework.

## Setup

Works only on Linux AMD64.

- [Install UV](https://github.com/astral-sh/uv/blob/main/README.md)

```sh
uv venv
```

```sh
source .venv/bin/activate
```

```sh
uv pip install -e '.[test]'
```

## Test

```sh
pytest
```
