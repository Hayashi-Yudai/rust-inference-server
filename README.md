# Rust Inference Server

A toy inference server written by Rust. The model that server has is trained by PyTorch with titanic dataset

## How to run

Build Docker container.

```console
./scripts/build.sh
```

Create model in container and run server

```console
./scripts/run.sh
cd /app/src/python-model
uv run src/main.py
cd /app/src/rust-server
cargo run --release &
./scripts/request.sh
```

The response is like the following

```json
{
  "message": "Died",
  "probability": 0.9892974495887756
}
```