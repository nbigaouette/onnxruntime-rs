# onnxruntime requires execinfo.h to build, which only works on glibc-based systems, so alpine is out...
FROM debian:bullseye-slim as base

RUN apt-get update && apt-get -y dist-upgrade

FROM base AS onnxruntime

RUN apt-get install -y \
    git \
    bash \
    python3 \
    cmake \
    git \
    build-essential \
    llvm \
    locales

# onnxruntime built in tests need en_US.UTF-8 available
# Uncomment en_US.UTF-8, then generate
RUN sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen && locale-gen

# build onnxruntime
RUN mkdir -p /opt/onnxruntime/tmp
# onnxruntime build relies on being in a git repo, so can't just get a tarball
# it's a big repo, so fetch shallowly
RUN cd /opt/onnxruntime/tmp && \
    git clone --recursive --depth 1 --shallow-submodules https://github.com/Microsoft/onnxruntime

# use version that onnxruntime-sys expects
RUN cd /opt/onnxruntime/tmp/onnxruntime && \
    git fetch --depth 1 origin tag v1.6.0 && \
    git checkout v1.6.0

RUN /opt/onnxruntime/tmp/onnxruntime/build.sh --config RelWithDebInfo --build_shared_lib --parallel

# Build ort-customops, linked against the onnxruntime built above.
# No tags / releases yet - that commit is from 2021-02-16
RUN mkdir -p /opt/ort-customops/tmp && \
    cd /opt/ort-customops/tmp && \
    git clone --recursive https://github.com/microsoft/ort-customops.git && \
    cd ort-customops && \
    git checkout 92f6b51106c9e9143c452e537cb5e41d2dcaa266

RUN cd /opt/ort-customops/tmp/ort-customops && \
    ./build.sh -D ONNXRUNTIME_LIB_DIR=/opt/onnxruntime/tmp/onnxruntime/build/Linux/RelWithDebInfo


# install rust toolchain
FROM base AS rust-toolchain

ARG RUST_VERSION=1.50.0

RUN apt-get install -y \
    curl

# install rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y  --default-toolchain $RUST_VERSION

ENV PATH $PATH:/root/.cargo/bin


# build onnxruntime-rs
FROM rust-toolchain as onnxruntime-rs
# clang & llvm needed by onnxruntime-sys
RUN apt-get install -y \
    build-essential \
    llvm-dev \
    libclang-dev \
    clang

RUN mkdir -p \
    /onnxruntime-rs/build/onnxruntime-sys/src/ \
    /onnxruntime-rs/build/onnxruntime/src/ \
    /onnxruntime-rs/build/onnxruntime/tests/ \
    /opt/onnxruntime/lib \
    /opt/ort-customops/lib

COPY --from=onnxruntime /opt/onnxruntime/tmp/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so /opt/onnxruntime/lib/
COPY --from=onnxruntime /opt/ort-customops/tmp/ort-customops/out/Linux/libortcustomops.so /opt/ort-customops/lib/

WORKDIR /onnxruntime-rs/build

ENV ORT_STRATEGY=system
# this has /lib/ appended to it and is used as a lib search path in onnxruntime-sys's build.rs
ENV ORT_LIB_LOCATION=/opt/onnxruntime/

ENV ONNXRUNTIME_RS_TEST_ORT_CUSTOMOPS_LIB=/opt/ort-customops/lib/libortcustomops.so

# create enough of an empty project that dependencies can build
COPY /Cargo.lock /Cargo.toml /onnxruntime-rs/build/
COPY /onnxruntime/Cargo.toml /onnxruntime-rs/build/onnxruntime/
COPY /onnxruntime-sys/Cargo.toml /onnxruntime-sys/build.rs /onnxruntime-rs/build/onnxruntime-sys/

CMD cargo test

# build dependencies and clean the bogus contents of our two packages
RUN touch \
        onnxruntime/src/lib.rs \
        onnxruntime/tests/integration_tests.rs \
        onnxruntime-sys/src/lib.rs \
    && cargo build --tests \
    && cargo clean --package onnxruntime-sys \
    && cargo clean --package onnxruntime \
    && rm -rf \
        onnxruntime/src/ \
        onnxruntime/tests/ \
        onnxruntime-sys/src/

# now build the actual source
COPY /test-models test-models
COPY /onnxruntime-sys/src onnxruntime-sys/src
COPY /onnxruntime/src onnxruntime/src
COPY /onnxruntime/tests onnxruntime/tests

RUN ln -s /opt/onnxruntime/lib/libonnxruntime.so /opt/onnxruntime/lib/libonnxruntime.so.1.6.0
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

RUN cargo build --tests
