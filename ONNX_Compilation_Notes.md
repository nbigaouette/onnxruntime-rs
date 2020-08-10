# Building ONNX Runtime

While ONNX requires cmake to build, bindgen (used to automatically build the bindings) requires
llvm:

```sh
brew install llvm cmake

# bindgen needs this to find llvm/clang
export LLVM_CONFIG_PATH=/usr/local/opt/llvm/bin/llvm-config
```

The `build.rs` script uses the `ONNXRUNTIME_INSTALL_DIR` environment variable to
find the built library and its headers. Make sure to point to the proper location:

```sh
❯ export ONNXRUNTIME_INSTALL_DIR=/full/path/to/onnxruntime
```

**NOTE**: The [`.cargo/config`](.cargo/config) file assumes the library is installed
          in the `target/onnxruntime` directory.

```sh
❯ git clone https://github.com/microsoft/onnxruntime.git onnxruntime.git
❯ cd onnxruntime.git
❯ git checkout v1.3.1
# Debug build with install directory inside our own 'target' directory
# Takes ~1/2 hour on a macbook pro 2.9 GHz 16 GB
❯ ./build.sh --config Debug --build_shared_lib --parallel --cmake_extra_defines="CMAKE_INSTALL_PREFIX=../../../../target/onnxruntime"
❯ cd build/Linux/Debug
❯ make install
```

---

**NOTE**: Cargo will link _dynamically_ to `libonnxruntime.{so,dylib}`. If the shared library is
located _outsite_ of the target directory, cargo will use an `rpath` instead of putting the full
path to the library in the built artifacts (see [this comment](https://github.com/rust-lang/cargo/issues/4421#issuecomment-325209304)).

For example:

```sh
❯ otool -L target/debug/examples/c_api_sample
target/debug/examples/c_api_sample:
        @rpath/libonnxruntime.1.3.1.dylib (compatibility version 0.0.0, current version 1.3.1)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)
        /usr/lib/libresolv.9.dylib (compatibility version 1.0.0, current version 1.0.0)
```

If `libonnxruntime.{so,dylib}` is not in a standard
directory (for example it was just compiled and not installed in a system directory) then
the library _will not be found_. For example:

```sh
❯ target/debug/examples/c_api_sample
dyld: Library not loaded: @rpath/libonnxruntime.1.3.1.dylib
  Referenced from: [...]/target/debug/examples/c_api_sample
  Reason: image not found
```

You _might_ need to use `DYLD_LIBRARY_PATH` (on macOS) to point to the directory where
`libonnxruntime.dylib` is located to execute a binary (or debug it). Note that this is not
the proper way as it messes _System Integrity Protection_.

Commit 4315b5a381faa11330fe554c69392f59 set the path in
[`.cargo/config`](.cargo/config) (for macOS) and should allow running the
binaries without setting an environment variable.

---
