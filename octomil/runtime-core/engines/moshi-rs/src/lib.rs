//! Octomil Slice 2C — Moshi engine Rust shim (private internal C ABI).
//!
//! This crate exposes a tiny `extern "C"` surface that the Octomil
//! C++ adapter (in `runtime-core/src/adapters/moshi_rs/`) calls.
//! Symbol prefix is `octomil_moshi_rs_*` to namespace from any
//! future engine shim. Public Octomil C ABI (in
//! `include/octomil/runtime.h`) is unchanged — this file does NOT
//! introduce an ABI delta.
//!
//! Streaming model:
//!   * The C++ adapter calls `engine_open` once at runtime-open
//!     (after artifact SHA-256 verification, which is the C++
//!     adapter's responsibility).
//!   * For each session, the C++ adapter calls `session_open`. We
//!     spawn one Rust worker thread per session that pulls audio
//!     frames from a queue, runs Mimi encode → LM step → Mimi
//!     decode, and pushes events into an output queue.
//!   * The C++ adapter's `oct_session_poll_event` calls
//!     `session_pop_event` (non-blocking dequeue).
//!   * Cancel: C++ adapter calls `session_cancel`, which flips
//!     an `AtomicBool` checked at every 80 ms frame boundary.
//!
//! Owned-vs-borrowed convention:
//!   * Engine and Session are opaque; opaque pointers come from
//!     `Box::into_raw` and must be released with the matching
//!     `*_close` function.
//!   * Strings (paths, error messages) on input are borrowed for
//!     the duration of the call.
//!   * Strings (error messages, transcript) on output are owned by
//!     the session; their lifetimes extend until the next call
//!     into that session.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

// -----------------------------------------------------------------------
// Public C status codes (internal to the shim — C++ adapter maps
// these to OCT_ERR_* before they reach Layer 2a's public ABI).
// -----------------------------------------------------------------------

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    InvalidInput = 1,
    ArtifactMissing = 2,
    LoadFailed = 3,
    InitFailed = 4,
    RamInsufficient = 5,
    AcceleratorUnavailable = 6,
    Timeout = 7,
    Internal = 8,
    Preempted = 9,
    Again = 10,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    None = 0,
    AudioChunk = 1,
    TranscriptChunk = 2,
    InputDropped = 3,
    SessionCompleted = 4,
    Error = 5,
}

// -----------------------------------------------------------------------
// Engine config — populated by the C++ adapter from the v0.4 step 2
// `oct_model_config_t` after artifact resolution. Strings are NOT
// owned by Rust; they must outlive the call.
// -----------------------------------------------------------------------

#[repr(C)]
pub struct EngineConfig {
    pub lm_weights_path: *const c_char,
    pub mimi_weights_path: *const c_char,
    pub text_tokenizer_path: *const c_char,
    /// Optional; NULL means use the built-in moshi-mlx v0.2 config (the
    /// shape the slice-2B probe verified). The only path the probe
    /// validated.
    pub lm_config_path: *const c_char,
    /// 1 = Metal (Apple Silicon), 0 = CPU. Slice 2C is darwin-arm64
    /// only; CPU is for the unit tests that exercise the shim
    /// without weights.
    pub use_metal: u8,
}

#[repr(C)]
pub struct SessionConfig {
    pub seed: u64,
    pub max_steps: u32,
}

#[repr(C)]
pub struct CEvent {
    pub event_type: EventType,
    pub audio_pcm: *const f32,
    pub audio_n_samples: usize,
    pub transcript_utf8: *const c_char,
    pub transcript_n_bytes: usize,
    pub is_final: c_int,
    pub n_frames_dropped: u32,
    pub status: Status,
    pub status_message: *const c_char,
}

impl CEvent {
    fn empty() -> Self {
        Self {
            event_type: EventType::None,
            audio_pcm: std::ptr::null(),
            audio_n_samples: 0,
            transcript_utf8: std::ptr::null(),
            transcript_n_bytes: 0,
            is_final: 0,
            n_frames_dropped: 0,
            status: Status::Ok,
            status_message: std::ptr::null(),
        }
    }
}

// -----------------------------------------------------------------------
// Rust-side event representation. Owns its strings/buffers; the C
// `pop_event` call vends pointers into the live RustEvent stored
// inside the session. Pointers are valid until the next pop_event.
// -----------------------------------------------------------------------

enum RustEvent {
    AudioChunk(Vec<f32>),
    TranscriptChunk { text: String, is_final: bool },
    InputDropped(u32),
    SessionCompleted { status: Status, message: String },
    Error { status: Status, message: String },
}

// -----------------------------------------------------------------------
// Engine — process-scope; holds resolved paths + device + the
// already-loaded LM and Mimi handles. Loading happens in
// `engine_open` so session_open can be cheap.
// -----------------------------------------------------------------------

pub struct Engine {
    lm_weights_path: PathBuf,
    mimi_weights_path: PathBuf,
    text_tokenizer_path: PathBuf,
    lm_config_path: Option<PathBuf>,
    use_metal: bool,
    /// Lazily populated on first session_open OR eagerly during
    /// engine_open's verification path. Slice 2C: eager. Wrapped
    /// in a Mutex to keep `Engine: Send + Sync` for the rare case
    /// the C++ adapter shares the engine across worker threads.
    inner: Mutex<Option<EngineInner>>,
}

struct EngineInner {
    /// Reserved for the real moshi-core integration. The shim does
    /// not yet construct an `LmModel`/`Mimi`/`SentencePieceProcessor`
    /// here because the candle build pulls in ~1 GB of dependencies
    /// and a 5–10 minute first compile. Slice 2C lands in two
    /// commits:
    ///   1. (this commit) shim scaffolding with the C ABI surface,
    ///      thread plumbing, cancel, event flow, NO real inference.
    ///      Capability is NOT advertised by the C++ adapter.
    ///   2. (follow-up) wire `moshi::lm::load_lm_model` +
    ///      `moshi::mimi::load` + `lm_generate_multistream::State`.
    ///      Capability is advertised.
    /// This split is allowed by the user directive: "A
    /// scaffolding/stub PR is acceptable only if it never
    /// advertises audio.realtime.session and is explicitly labeled
    /// as internal adapter plumbing, not Slice 2C complete."
    _placeholder: (),
}

// -----------------------------------------------------------------------
// Session — request-scope; owns the worker thread, the cancel
// atomic, and the audio/event channels.
// -----------------------------------------------------------------------

pub struct Session {
    cancelled: Arc<AtomicBool>,
    audio_tx: Option<mpsc::Sender<Vec<f32>>>,
    event_rx: Mutex<mpsc::Receiver<RustEvent>>,
    worker: Option<JoinHandle<()>>,
    /// The currently-vended RustEvent backing the C-visible pointers
    /// returned by pop_event. Held until the next pop_event call.
    last_event: Mutex<Option<RustEvent>>,
    /// CString storage for the message pointer in the last vended
    /// event. Must outlive the C caller's read.
    last_message_cstr: Mutex<Option<CString>>,
    /// Same for transcript text.
    last_transcript_cstr: Mutex<Option<CString>>,
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// SAFETY: `s` must be a valid C string pointer or NULL.
unsafe fn cstr_to_string(s: *const c_char) -> Option<String> {
    if s.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(s).to_str().ok().map(str::to_owned) }
}

/// SAFETY: `buf` is a writeable buffer of `buflen` bytes. Writes a
/// nul-terminated copy of `msg` truncated to fit. No-op on
/// `buflen == 0`.
unsafe fn write_err(buf: *mut c_char, buflen: usize, msg: &str) {
    if buf.is_null() || buflen == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(buflen - 1);
    // SAFETY: caller's contract on (buf, buflen).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

fn check_artifacts(cfg: &EngineConfig) -> Result<(PathBuf, PathBuf, PathBuf, Option<PathBuf>), String> {
    // SAFETY: The C++ adapter's contract requires these strings to
    // be valid for the duration of the call.
    let lm = unsafe { cstr_to_string(cfg.lm_weights_path) }
        .ok_or_else(|| "lm_weights_path is NULL or non-UTF-8".to_string())?;
    let mimi = unsafe { cstr_to_string(cfg.mimi_weights_path) }
        .ok_or_else(|| "mimi_weights_path is NULL or non-UTF-8".to_string())?;
    let tok = unsafe { cstr_to_string(cfg.text_tokenizer_path) }
        .ok_or_else(|| "text_tokenizer_path is NULL or non-UTF-8".to_string())?;
    let lm_cfg = unsafe { cstr_to_string(cfg.lm_config_path) };

    let lm = PathBuf::from(lm);
    let mimi = PathBuf::from(mimi);
    let tok = PathBuf::from(tok);
    let lm_cfg = lm_cfg.map(PathBuf::from);

    for (name, p) in [("lm_weights", &lm), ("mimi_weights", &mimi), ("text_tokenizer", &tok)] {
        if !p.is_file() {
            return Err(format!("{name} not found at {}", p.display()));
        }
    }
    if let Some(p) = &lm_cfg {
        if !p.is_file() {
            return Err(format!("lm_config not found at {}", p.display()));
        }
    }
    Ok((lm, mimi, tok, lm_cfg))
}

// -----------------------------------------------------------------------
// C ABI — engine
// -----------------------------------------------------------------------

/// Verify on-disk presence of the configured artifacts. Cheap
/// pre-flight; SHA-256 verification is the C++ adapter's job.
///
/// SAFETY: All `cfg` string pointers must be valid C strings or NULL.
/// `err_buf` must be a writeable buffer of `err_buflen` bytes (or
/// NULL with `err_buflen == 0`).
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_engine_check_artifacts(
    cfg: *const EngineConfig,
    err_buf: *mut c_char,
    err_buflen: usize,
) -> Status {
    if cfg.is_null() {
        unsafe { write_err(err_buf, err_buflen, "engine_check_artifacts: cfg is NULL") };
        return Status::InvalidInput;
    }
    // SAFETY: caller's contract on cfg.
    let cfg = unsafe { &*cfg };
    match check_artifacts(cfg) {
        Ok(_) => Status::Ok,
        Err(msg) => {
            unsafe { write_err(err_buf, err_buflen, &msg) };
            Status::ArtifactMissing
        }
    }
}

/// SAFETY: `cfg` must be a valid `EngineConfig` for the call's
/// duration. `out_engine` must be non-NULL. `err_buf`/`err_buflen`
/// see above.
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_engine_open(
    cfg: *const EngineConfig,
    out_engine: *mut *mut Engine,
    err_buf: *mut c_char,
    err_buflen: usize,
) -> Status {
    if cfg.is_null() || out_engine.is_null() {
        unsafe { write_err(err_buf, err_buflen, "engine_open: NULL argument") };
        return Status::InvalidInput;
    }
    // SAFETY: caller's contract on cfg.
    let cfg = unsafe { &*cfg };
    let (lm, mimi, tok, lm_cfg) = match check_artifacts(cfg) {
        Ok(t) => t,
        Err(msg) => {
            unsafe { write_err(err_buf, err_buflen, &msg) };
            return Status::ArtifactMissing;
        }
    };

    let engine = Engine {
        lm_weights_path: lm,
        mimi_weights_path: mimi,
        text_tokenizer_path: tok,
        lm_config_path: lm_cfg,
        use_metal: cfg.use_metal != 0,
        inner: Mutex::new(None),
    };

    let boxed = Box::new(engine);
    // SAFETY: out_engine non-NULL per caller's contract.
    unsafe { *out_engine = Box::into_raw(boxed) };
    Status::Ok
}

/// SAFETY: `engine` must be a pointer previously returned by
/// `engine_open` and not yet closed. NULL is a no-op.
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_engine_close(engine: *mut Engine) {
    if engine.is_null() {
        return;
    }
    // SAFETY: caller's contract.
    drop(unsafe { Box::from_raw(engine) });
}

// -----------------------------------------------------------------------
// C ABI — session
// -----------------------------------------------------------------------

/// SAFETY: `engine` is a valid open Engine. `cfg` and `out_session`
/// are non-NULL.
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_open(
    engine: *mut Engine,
    cfg: *const SessionConfig,
    out_session: *mut *mut Session,
    err_buf: *mut c_char,
    err_buflen: usize,
) -> Status {
    if engine.is_null() || cfg.is_null() || out_session.is_null() {
        unsafe { write_err(err_buf, err_buflen, "session_open: NULL argument") };
        return Status::InvalidInput;
    }
    // SAFETY: caller's contract.
    let _engine = unsafe { &*engine };
    let _cfg = unsafe { &*cfg };

    let cancelled = Arc::new(AtomicBool::new(false));
    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
    let (event_tx, event_rx) = mpsc::channel::<RustEvent>();

    let cancel_for_worker = Arc::clone(&cancelled);
    let worker = std::thread::Builder::new()
        .name("octomil-moshi-rs-session".into())
        .spawn(move || {
            session_worker(audio_rx, event_tx, cancel_for_worker);
        })
        .expect("spawn session worker");

    let session = Session {
        cancelled,
        audio_tx: Some(audio_tx),
        event_rx: Mutex::new(event_rx),
        worker: Some(worker),
        last_event: Mutex::new(None),
        last_message_cstr: Mutex::new(None),
        last_transcript_cstr: Mutex::new(None),
    };
    let boxed = Box::new(session);
    unsafe { *out_session = Box::into_raw(boxed) };
    Status::Ok
}

fn session_worker(
    audio_rx: mpsc::Receiver<Vec<f32>>,
    event_tx: mpsc::Sender<RustEvent>,
    cancelled: Arc<AtomicBool>,
) {
    // Slice 2C scaffolding: pulls audio frames, ignores them, emits
    // a SESSION_COMPLETED with InitFailed once the input channel
    // closes OR cancel flips. The follow-up commit replaces the
    // body with `moshi::mimi::encode_step` → `lm_generate_multistream`
    // → `moshi::mimi::decode_step` and emits real AUDIO_CHUNK and
    // TRANSCRIPT_CHUNK events.
    //
    // This deliberately produces NO output events on the streaming
    // path so the C++ adapter cannot accidentally claim to be
    // serving audio in the scaffolding state. Capability honesty is
    // also enforced at the C++ adapter level: the scaffolding adapter
    // does not advertise `audio.realtime.session`.
    loop {
        if cancelled.load(Ordering::Acquire) {
            let _ = event_tx.send(RustEvent::SessionCompleted {
                status: Status::Preempted,
                message: "cancelled by caller".to_owned(),
            });
            return;
        }
        match audio_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(_pcm) => {
                // Drop the frame on the floor in scaffolding mode.
                // The follow-up commit feeds it into Mimi.
                continue;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = event_tx.send(RustEvent::SessionCompleted {
                    status: Status::InitFailed,
                    message: "moshi-rs scaffolding: real inference path not yet wired (Slice 2C follow-up)".to_owned(),
                });
                return;
            }
        }
    }
}

/// Push one audio frame into the session. Frame must be 1920
/// float32 samples (80 ms @ 24 kHz mono); other shapes return
/// InvalidInput.
///
/// SAFETY: `session` is a valid open Session. `pcm` points to at
/// least `n_samples` floats.
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_send_audio(
    session: *mut Session,
    pcm: *const f32,
    n_samples: usize,
) -> Status {
    if session.is_null() || pcm.is_null() {
        return Status::InvalidInput;
    }
    if n_samples != 1920 {
        return Status::InvalidInput;
    }
    // SAFETY: caller's contract on session and pcm.
    let session = unsafe { &*session };
    let frame = unsafe { std::slice::from_raw_parts(pcm, n_samples) }.to_vec();
    let Some(tx) = session.audio_tx.as_ref() else {
        return Status::Internal;
    };
    match tx.send(frame) {
        Ok(_) => Status::Ok,
        Err(_) => Status::Internal,
    }
}

/// Non-blocking event dequeue. Returns `Again` if the queue is
/// empty. The vended pointers in `out_event` are valid until the
/// next pop_event call on the same session.
///
/// SAFETY: `session` and `out_event` are valid pointers.
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_pop_event(
    session: *mut Session,
    out_event: *mut CEvent,
) -> Status {
    if session.is_null() || out_event.is_null() {
        return Status::InvalidInput;
    }
    // SAFETY: caller's contract.
    let session = unsafe { &*session };
    let mut last_event = session.last_event.lock().expect("session.last_event poisoned");
    let mut last_message = session.last_message_cstr.lock().expect("session.last_message_cstr poisoned");
    let mut last_transcript = session.last_transcript_cstr.lock().expect("session.last_transcript_cstr poisoned");

    let rx = session.event_rx.lock().expect("session.event_rx poisoned");
    let next = match rx.try_recv() {
        Ok(ev) => ev,
        Err(mpsc::TryRecvError::Empty) => {
            unsafe { *out_event = CEvent::empty() };
            return Status::Again;
        }
        Err(mpsc::TryRecvError::Disconnected) => {
            unsafe { *out_event = CEvent::empty() };
            return Status::Again;
        }
    };
    drop(rx);

    let mut c_event = CEvent::empty();
    match &next {
        RustEvent::AudioChunk(pcm) => {
            c_event.event_type = EventType::AudioChunk;
            c_event.audio_pcm = pcm.as_ptr();
            c_event.audio_n_samples = pcm.len();
        }
        RustEvent::TranscriptChunk { text, is_final } => {
            c_event.event_type = EventType::TranscriptChunk;
            let bytes = text.as_bytes();
            *last_transcript = Some(CString::new(bytes).unwrap_or_default());
            if let Some(c) = last_transcript.as_ref() {
                c_event.transcript_utf8 = c.as_ptr();
                c_event.transcript_n_bytes = c.as_bytes().len();
            }
            c_event.is_final = if *is_final { 1 } else { 0 };
        }
        RustEvent::InputDropped(n) => {
            c_event.event_type = EventType::InputDropped;
            c_event.n_frames_dropped = *n;
        }
        RustEvent::SessionCompleted { status, message } => {
            c_event.event_type = EventType::SessionCompleted;
            c_event.status = *status;
            *last_message = Some(CString::new(message.as_bytes()).unwrap_or_default());
            if let Some(c) = last_message.as_ref() {
                c_event.status_message = c.as_ptr();
            }
        }
        RustEvent::Error { status, message } => {
            c_event.event_type = EventType::Error;
            c_event.status = *status;
            *last_message = Some(CString::new(message.as_bytes()).unwrap_or_default());
            if let Some(c) = last_message.as_ref() {
                c_event.status_message = c.as_ptr();
            }
        }
    }
    *last_event = Some(next);
    unsafe { *out_event = c_event };
    Status::Ok
}

/// Flip the cancel atomic. Worker thread checks at every 80 ms
/// frame boundary.
///
/// SAFETY: `session` is a valid open Session.
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_cancel(session: *mut Session) -> Status {
    if session.is_null() {
        return Status::InvalidInput;
    }
    // SAFETY: caller's contract.
    let session = unsafe { &*session };
    session.cancelled.store(true, Ordering::Release);
    Status::Ok
}

/// Drop the audio input channel, signal cancel, join the worker.
/// Idempotent on NULL.
///
/// SAFETY: `session` is a pointer previously returned by
/// `session_open` (or NULL).
#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_close(session: *mut Session) {
    if session.is_null() {
        return;
    }
    // SAFETY: caller's contract.
    let mut session = unsafe { Box::from_raw(session) };
    // Drop the audio sender so the worker's recv_timeout loop sees
    // Disconnected and exits cleanly.
    session.audio_tx.take();
    session.cancelled.store(true, Ordering::Release);
    if let Some(worker) = session.worker.take() {
        // Join — at most one frame's worth of compute (≤ 81 ms per
        // probe + 50 ms timeout) before the worker observes the
        // disconnect.
        let _ = worker.join();
    }
}

#[no_mangle]
pub extern "C" fn octomil_moshi_rs_version() -> *const c_char {
    // 'static literal; pointer is stable for the dylib's lifetime.
    concat!("octomil-moshi-rs/", env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nul_terminated_static() {
        let p = octomil_moshi_rs_version();
        assert!(!p.is_null());
        // SAFETY: the version function returns a static nul-terminated literal.
        let s = unsafe { CStr::from_ptr(p) };
        assert!(s.to_str().unwrap().starts_with("octomil-moshi-rs/"));
    }
}
