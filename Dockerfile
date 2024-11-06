FROM ubuntu:latest

WORKDIR /app

RUN apt update && apt install -y curl wget unzip gcc g++ make


# Install Rust
ENV PATH="$PATH:/root/.cargo/bin"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && echo 'source $HOME/.cargo/env' >> ~/.bashrc

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && uv python install 3.12

# Install libtorch
# ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LIBTORCH=/app/libtorch
ENV LIBTORCH_INCLUDE=/app/libtorch
ENV LIBTORCH_LIB=/app/libtorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.0%2Bcpu.zip \
  && unzip libtorch-shared-with-deps-2.5.0+cpu.zip \
  && rm libtorch-shared-with-deps-2.5.0+cpu.zip
