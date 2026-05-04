#
# liboctomil-runtime — Layer 2a native runtime core
#
# Slice-2 deliverable per strategy/runtime-architecture-v2.md. This
# CMakeLists builds the C ABI dylib defined by include/octomil/runtime.h.
#
# Slice-1 (already merged in PR #508) shipped only the header. This
# slice ships:
#   * The build system (this file + BUILD.md).
#   * Stub implementations of every OCT_API symbol — every entry
#     point returns OCT_STATUS_UNSUPPORTED until slice-2 lands the
#     Moshi-on-MLX session adapter.
#   * A smoke test that loads the dylib and asserts version /
#     last-error / capabilities behave correctly.
#
# The stubs let downstream slices (Python cffi binding in slice 3, the
# iOS framework in slice 4) start integrating against a real binary
# while the runtime core is being filled in. Bindings exercise the
# full ABI surface; backend-specific tests are gated on the engine
# being implemented.
#

cmake_minimum_required(VERSION 3.20)
project(octomil_runtime
    VERSION 0.1.0
    DESCRIPTION "Octomil native runtime core (Layer 2a)"
    LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Strict export-by-default-OFF so OCT_API decoration is the source of
# truth for the public surface. Anything not OCT_API'd stays internal.
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(OCT_BUILD_TESTS "Build the runtime ABI smoke tests" ON)
option(OCT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# Slice 2C engine adapter — Moshi via the Rust shim around moshi-core.
# Default ON for darwin-arm64 (the only platform Moshi runs on
# natively today); OFF elsewhere because the Rust crate doesn't
# build for non-Metal targets in this slice.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(OCT_ENABLE_ENGINE_MOSHI_RS_DEFAULT ON)
else()
    set(OCT_ENABLE_ENGINE_MOSHI_RS_DEFAULT OFF)
endif()
option(OCT_ENABLE_ENGINE_MOSHI_RS "Build the Moshi-rs engine adapter (darwin-arm64)" ${OCT_ENABLE_ENGINE_MOSHI_RS_DEFAULT})

# -----------------------------------------------------------------------------
# Library target — liboctomil-runtime
# -----------------------------------------------------------------------------
add_library(octomil_runtime SHARED
    src/runtime.cpp
    src/adapters/adapter_base.cpp
)
if(OCT_ENABLE_ENGINE_MOSHI_RS)
    target_sources(octomil_runtime PRIVATE
        src/adapters/moshi_rs/moshi_rs_adapter.cpp
    )
endif()

target_include_directories(octomil_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(octomil_runtime
    PRIVATE
        # Match the header's Windows export gate. The header at
        # include/octomil/runtime.h uses OCTOMIL_RUNTIME_BUILDING to
        # switch __declspec(dllexport) / __declspec(dllimport); without
        # it the MSVC build sees exported function definitions as
        # dllimport and produces an unlinkable DLL.
        OCTOMIL_RUNTIME_BUILDING=1
)

if(OCT_ENABLE_ENGINE_MOSHI_RS)
    target_compile_definitions(octomil_runtime PRIVATE
        OCT_ENABLE_ENGINE_MOSHI_RS=1
        OCT_HAVE_MOSHI_RS_SHIM=1
    )
endif()

# -----------------------------------------------------------------------------
# Moshi-rs Rust shim — built via cargo, linked as a static archive.
# This is intentionally a thin add_custom_command + IMPORTED target;
# upgrading to corrosion-rs is a follow-up if/when more Rust shims
# land.
# -----------------------------------------------------------------------------
if(OCT_ENABLE_ENGINE_MOSHI_RS)
    set(MOSHI_RS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/engines/moshi-rs)
    set(MOSHI_RS_LIB ${MOSHI_RS_DIR}/target/release/liboctomil_moshi_rs.a)

    find_program(CARGO_EXECUTABLE cargo
        PATHS $ENV{HOME}/.cargo/bin /opt/homebrew/opt/rustup/bin
        NO_CACHE
    )
    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR
            "OCT_ENABLE_ENGINE_MOSHI_RS=ON but cargo not found. "
            "Install rustup (`brew install rustup && rustup default stable`) "
            "or set OCT_ENABLE_ENGINE_MOSHI_RS=OFF.")
    endif()

    add_custom_command(
        OUTPUT ${MOSHI_RS_LIB}
        COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path=${MOSHI_RS_DIR}/Cargo.toml
        WORKING_DIRECTORY ${MOSHI_RS_DIR}
        DEPENDS
            ${MOSHI_RS_DIR}/Cargo.toml
            ${MOSHI_RS_DIR}/src/lib.rs
        COMMENT "Building octomil-moshi-rs Rust shim (cargo build --release)"
        VERBATIM
    )
    add_custom_target(moshi_rs_static DEPENDS ${MOSHI_RS_LIB})
    add_dependencies(octomil_runtime moshi_rs_static)

    target_include_directories(octomil_runtime PRIVATE
        ${MOSHI_RS_DIR}/include
    )
    target_link_libraries(octomil_runtime PRIVATE ${MOSHI_RS_LIB})
    target_link_libraries(octomil_runtime PRIVATE
        # candle-metal pulls in CoreFoundation, Metal, and the
        # objc runtime via its dependencies; link them explicitly
        # so the static archive resolves at link time.
        "-framework CoreFoundation"
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Accelerate"
    )
endif()

if(MSVC)
    target_compile_options(octomil_runtime PRIVATE /W4)
    if(OCT_WARNINGS_AS_ERRORS)
        target_compile_options(octomil_runtime PRIVATE /WX)
    endif()
else()
    target_compile_options(octomil_runtime PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
    if(OCT_WARNINGS_AS_ERRORS)
        target_compile_options(octomil_runtime PRIVATE -Werror)
    endif()
endif()

set_target_properties(octomil_runtime PROPERTIES
    # Final library name on disk: liboctomil-runtime.{dylib,so,dll}
    OUTPUT_NAME "octomil-runtime"
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

# Public symbol list: enforce that ONLY OCT_API decorated symbols are
# exported. Linker-level export filtering is configured per-platform
# in scripts/exports/* if/when needed; today the visibility=hidden +
# OCT_API attribute combination is sufficient.

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(OCT_BUILD_TESTS)
    enable_testing()
    add_executable(test_abi_smoke tests/test_abi_smoke.cpp)
    target_link_libraries(test_abi_smoke PRIVATE octomil_runtime)
    target_include_directories(test_abi_smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME abi_smoke COMMAND test_abi_smoke)

    if(OCT_ENABLE_ENGINE_MOSHI_RS)
        add_executable(test_engine_capability_honesty tests/test_engine_capability_honesty.cpp)
        target_link_libraries(test_engine_capability_honesty PRIVATE octomil_runtime)
        target_include_directories(test_engine_capability_honesty PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        add_test(NAME engine_capability_honesty COMMAND test_engine_capability_honesty)
    endif()
endif()

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS octomil_runtime
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/octomil
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
