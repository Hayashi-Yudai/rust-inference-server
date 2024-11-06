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
ENV LIBTORCH=/app/libtorch
ENV LIBTORCH_INCLUDE=/app/libtorch
ENV LIBTORCH_LIB=/app/libtorch
ENV LD_LIBRARY_PATH=/app/libtorch/lib:$LD_LIBRARY_PATH

# For CPU
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip \
  && unzip libtorch-cxx11-abi-shared-with-deps-2.5.0+cpu.zip \
  && rm libtorch-cxx11-abi-shared-with-deps-2.5.0+cpu.zip
