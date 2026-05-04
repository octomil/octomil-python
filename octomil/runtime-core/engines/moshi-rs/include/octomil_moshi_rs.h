/*
 * Octomil Slice 2C — Moshi engine Rust shim (private internal C ABI).
 *
 * THIS HEADER IS NOT PART OF THE PUBLIC OCTOMIL ABI. It is the
 * internal contract between liboctomil-runtime's C++ adapter
 * (src/adapters/moshi_rs/) and the Rust shim that wraps upstream
 * `moshi-core`. SDKs and Layer-2b consumers MUST NOT include it.
 *
 * The public Octomil ABI is `include/octomil/runtime.h` (v0.4 step 2)
 * and is unchanged by this slice.
 *
 * Status code mapping into the public OCT_ERR_* taxonomy is the
 * responsibility of the C++ adapter (see
 * src/adapters/moshi_rs/moshi_rs_adapter.cpp). The shim's status
 * enum is INTERNAL and may grow without an Octomil ABI bump.
 */
#ifndef OCTOMIL_INTERNAL_MOSHI_RS_H
#define OCTOMIL_INTERNAL_MOSHI_RS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OCTOMIL_MOSHI_OK = 0,
    OCTOMIL_MOSHI_ERR_INVALID_INPUT = 1,
    OCTOMIL_MOSHI_ERR_ARTIFACT_MISSING = 2,
    OCTOMIL_MOSHI_ERR_LOAD_FAILED = 3,
    OCTOMIL_MOSHI_ERR_INIT_FAILED = 4,
    OCTOMIL_MOSHI_ERR_RAM_INSUFFICIENT = 5,
    OCTOMIL_MOSHI_ERR_ACCELERATOR_UNAVAILABLE = 6,
    OCTOMIL_MOSHI_ERR_TIMEOUT = 7,
    OCTOMIL_MOSHI_ERR_INTERNAL = 8,
    OCTOMIL_MOSHI_ERR_PREEMPTED = 9,
    OCTOMIL_MOSHI_ERR_AGAIN = 10,
} octomil_moshi_status_t;

typedef enum {
    OCTOMIL_MOSHI_EVENT_NONE = 0,
    OCTOMIL_MOSHI_EVENT_AUDIO_CHUNK = 1,
    OCTOMIL_MOSHI_EVENT_TRANSCRIPT_CHUNK = 2,
    OCTOMIL_MOSHI_EVENT_INPUT_DROPPED = 3,
    OCTOMIL_MOSHI_EVENT_SESSION_COMPLETED = 4,
    OCTOMIL_MOSHI_EVENT_ERROR = 5,
} octomil_moshi_event_type_t;

typedef struct octomil_moshi_engine octomil_moshi_engine_t;
typedef struct octomil_moshi_session octomil_moshi_session_t;

typedef struct {
    const char* lm_weights_path;
    const char* mimi_weights_path;
    const char* text_tokenizer_path;
    /* Optional; NULL means use the built-in moshi-mlx v0.2 config. */
    const char* lm_config_path;
    /* 1 = Metal (Apple Silicon), 0 = CPU. */
    uint8_t use_metal;
} octomil_moshi_engine_config_t;

typedef struct {
    uint64_t seed;
    uint32_t max_steps;
} octomil_moshi_session_config_t;

typedef struct {
    octomil_moshi_event_type_t event_type;
    /* AUDIO_CHUNK */
    const float* audio_pcm;
    size_t audio_n_samples;
    /* TRANSCRIPT_CHUNK */
    const char* transcript_utf8;
    size_t transcript_n_bytes;
    int is_final;
    /* INPUT_DROPPED */
    uint32_t n_frames_dropped;
    /* SESSION_COMPLETED / ERROR */
    octomil_moshi_status_t status;
    const char* status_message;
} octomil_moshi_event_t;

octomil_moshi_status_t octomil_moshi_rs_engine_check_artifacts(
    const octomil_moshi_engine_config_t* cfg,
    char* err_buf,
    size_t err_buflen
);

octomil_moshi_status_t octomil_moshi_rs_engine_open(
    const octomil_moshi_engine_config_t* cfg,
    octomil_moshi_engine_t** out_engine,
    char* err_buf,
    size_t err_buflen
);

void octomil_moshi_rs_engine_close(octomil_moshi_engine_t* engine);

octomil_moshi_status_t octomil_moshi_rs_session_open(
    octomil_moshi_engine_t* engine,
    const octomil_moshi_session_config_t* cfg,
    octomil_moshi_session_t** out_session,
    char* err_buf,
    size_t err_buflen
);

octomil_moshi_status_t octomil_moshi_rs_session_send_audio(
    octomil_moshi_session_t* session,
    const float* pcm,
    size_t n_samples
);

octomil_moshi_status_t octomil_moshi_rs_session_pop_event(
    octomil_moshi_session_t* session,
    octomil_moshi_event_t* out_event
);

octomil_moshi_status_t octomil_moshi_rs_session_cancel(octomil_moshi_session_t* session);

void octomil_moshi_rs_session_close(octomil_moshi_session_t* session);

const char* octomil_moshi_rs_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OCTOMIL_INTERNAL_MOSHI_RS_H */
