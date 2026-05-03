# Building `liboctomil-runtime`

## Prereqs

- CMake ≥ 3.20
- A C++17 toolchain
  - macOS: Xcode CLT (`xcode-select --install`) — Apple Clang is fine.
  - Linux: GCC 9+ or Clang 10+.
  - Windows: MSVC 19.30+ (Visual Studio 2022).
- Ninja is recommended but not required (`brew install ninja`).

## Build

```sh
cd octomil-python/octomil/runtime-core
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Outputs:

| Platform | Library file                         |
| -------- | ------------------------------------ |
| macOS    | `build/liboctomil-runtime.dylib`     |
| Linux    | `build/liboctomil-runtime.so`        |
| Windows  | `build/octomil-runtime.dll` + `.lib` |

The header is at `include/octomil/runtime.h`. Bindings (Python cffi
in slice 3, Swift in slice 4) consume both.

## Run the smoke test

```sh
cd octomil-python/octomil/runtime-core
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected:

```
Test project .../build
    Start 1: abi_smoke
1/1 Test #1: abi_smoke ........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1
```

The smoke test exercises every OCT_API entry point and asserts the
version-handshake / last-error contract. It does NOT test session
behavior — those tests live in slice 3 against the cffi binding +
the Python conformance suite.

## CMake options

| Option                   | Default | Effect                                   |
| ------------------------ | ------- | ---------------------------------------- |
| `OCT_BUILD_TESTS`        | `ON`    | Build + register `test_abi_smoke`.       |
| `OCT_WARNINGS_AS_ERRORS` | `ON`    | `-Werror` / `/WX`. Disable when probing. |

## What this slice ships

This is the **slice-2 build-system PR** per
`strategy/runtime-architecture-v2.md`. It ships:

- `CMakeLists.txt` — the build configuration.
- `src/runtime.cpp` — **stub** implementations of every OCT_API entry
  point. Version inspection + `oct_runtime_open` + `oct_runtime_close`
  - last-error helpers behave for real; everything else returns
    `OCT_STATUS_UNSUPPORTED` with a descriptive last-error message.
- `tests/test_abi_smoke.cpp` — verifies the symbols are exported and
  the contract above holds.

The stubs unblock downstream slices (Python cffi binding in slice 3,
Swift framework in slice 4) — bindings can start compiling against a
real `liboctomil-runtime.{dylib,so,dll}` while the actual session
adapter is being filled in. Slice-2 implementation replaces these
stubs file by file, starting with the Moshi-on-MLX session adapter
on macOS.

## Hard cutover policy

Per the strategy doc: when the runtime core takes over a capability
from the Python kernel, the Python implementation is removed in the
same release. No back-compat aliases. The Python conformance tests
stay; their backend switches to the native runtime via cffi.

## Cross-compile

Out of scope for slice-2. iOS framework packaging lands in slice 4;
Android JNI / WASM bindings track contract version per the migration
plan.
