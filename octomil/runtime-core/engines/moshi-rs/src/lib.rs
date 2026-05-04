//! Octomil Slice 2C — Moshi engine Rust shim (private internal C ABI).
//!
//! Wraps upstream `moshi-core` (kyutai-labs/moshi @ 0.6.4) with the
//! candle-rs `metal` backend on Apple Silicon. The C++ adapter in
//! `runtime-core/src/adapters/moshi_rs/` is the only consumer; SDKs
//! and Layer-2b never see this surface.
//!
//! Lifetime model:
//!   * `Engine`: process-scope. Holds device + dtype + resolved
//!     artifact paths + sentencepiece tokenizer + lm config. Cheap.
//!   * `Session`: request-scope. The session_worker thread loads
//!     the LM and Mimi from the engine's paths, runs the streaming
//!     loop, and drops them on close. One concurrent session is
//!     supported in Slice 2C; multi-session is Slice 3b's domain.
//!
//! Cancel: `Session.cancelled` is an AtomicBool checked at every
//! 80-ms frame boundary inside the worker. session_cancel flips it.
//! session_close additionally drops the audio sender (signaling
//! graceful exit) and joins the worker.

#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use moshi::lm_generate_multistream;
use moshi::{lm, mimi};

// -----------------------------------------------------------------------
// Public C status / event enums
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

#[repr(C)]
pub struct EngineConfig {
    pub lm_weights_path: *const c_char,
    pub mimi_weights_path: *const c_char,
    pub text_tokenizer_path: *const c_char,
    /// Optional; NULL = built-in `Config::v0_1_streaming(8)`.
    pub lm_config_path: *const c_char,
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
// Internal Rust event queue
// -----------------------------------------------------------------------

enum RustEvent {
    AudioChunk(Vec<f32>),
    TranscriptChunk { text: String, is_final: bool },
    InputDropped(u32),
    SessionCompleted { status: Status, message: String },
    Error { status: Status, message: String },
}

// -----------------------------------------------------------------------
// Engine — holds only the lightweight, share-across-sessions state.
// LM and Mimi are loaded in the session worker so each session gets
// its own KV cache + Mimi state without locking.
// -----------------------------------------------------------------------

pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    paths: ResolvedPaths,
    use_metal: bool,
    /// Loaded once at engine_open; cloned by reference into each
    /// session's tokenizer. SentencePieceProcessor is `Sync`.
    text_tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    lm_cfg: lm::Config,
    generated_audio_codebooks: usize,
    /// Slice 2C: one session at a time. The lock is acquired by
    /// session_open and released on session_close.
    session_lock: Mutex<()>,
}

#[derive(Clone)]
struct ResolvedPaths {
    lm_weights: PathBuf,
    mimi_weights: PathBuf,
    text_tokenizer: PathBuf,
    lm_config: Option<PathBuf>,
}

fn open_device(use_metal: bool) -> Result<Device> {
    if use_metal {
        Device::new_metal(0).context("candle: failed to open Metal device 0")
    } else {
        Ok(Device::Cpu)
    }
}

fn read_lm_config(path: Option<&PathBuf>) -> Result<lm::Config> {
    match path {
        Some(p) => {
            let s = std::fs::read_to_string(p)
                .with_context(|| format!("read lm_config at {}", p.display()))?;
            let trimmed = s.trim_start();
            if trimmed.starts_with('{') {
                serde_json::from_str(&s).context("parse lm_config as JSON")
            } else {
                toml::from_str(&s).context("parse lm_config as TOML")
            }
        }
        None => Ok(lm::Config::v0_1_streaming(8)),
    }
}

// -----------------------------------------------------------------------
// Session
// -----------------------------------------------------------------------

pub struct Session {
    cancelled: Arc<AtomicBool>,
    audio_tx: Option<mpsc::Sender<Vec<f32>>>,
    event_rx: Mutex<mpsc::Receiver<RustEvent>>,
    worker: Option<JoinHandle<()>>,
    last_event: Mutex<Option<RustEvent>>,
    last_message_cstr: Mutex<Option<CString>>,
    last_transcript_cstr: Mutex<Option<CString>>,
    /// Held for the session's lifetime so a second session_open
    /// while one is active fails loudly. Released in session_close
    /// via drop order.
    _engine_session_lock: Option<std::sync::MutexGuard<'static, ()>>,
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

unsafe fn cstr_to_string(s: *const c_char) -> Option<String> {
    if s.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(s).to_str().ok().map(str::to_owned) }
}

unsafe fn write_err(buf: *mut c_char, buflen: usize, msg: &str) {
    if buf.is_null() || buflen == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(buflen - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

fn check_artifacts(cfg: &EngineConfig) -> Result<ResolvedPaths, String> {
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
    Ok(ResolvedPaths {
        lm_weights: lm,
        mimi_weights: mimi,
        text_tokenizer: tok,
        lm_config: lm_cfg,
    })
}

// -----------------------------------------------------------------------
// Streaming worker
// -----------------------------------------------------------------------

const FRAME_SAMPLES: usize = 1920; // 80 ms @ 24 kHz

fn session_worker(
    engine: Arc<EngineInner>,
    sess_cfg: SessionConfig,
    audio_rx: mpsc::Receiver<Vec<f32>>,
    event_tx: mpsc::Sender<RustEvent>,
    cancelled: Arc<AtomicBool>,
) {
    if let Err(e) = run_session(engine, sess_cfg, audio_rx, event_tx.clone(), cancelled.clone()) {
        let msg = format!("{e:#}");
        let _ = event_tx.send(RustEvent::Error {
            status: Status::Internal,
            message: msg.clone(),
        });
        let _ = event_tx.send(RustEvent::SessionCompleted {
            status: Status::Internal,
            message: msg,
        });
    }
}

fn run_session(
    engine: Arc<EngineInner>,
    sess_cfg: SessionConfig,
    audio_rx: mpsc::Receiver<Vec<f32>>,
    event_tx: mpsc::Sender<RustEvent>,
    cancelled: Arc<AtomicBool>,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    eprintln!("[moshi-rs] worker: opening device (use_metal={})", engine.use_metal);
    let device = open_device(engine.use_metal)?;
    let dtype = device.bf16_default_to_f32();
    eprintln!("[moshi-rs] worker: device ok in {:?}, dtype={:?}", t0.elapsed(), dtype);

    let t1 = std::time::Instant::now();
    let mut mimi_model = mimi::load(
        engine.paths.mimi_weights.to_string_lossy().as_ref(),
        Some(8),
        &device,
    )
    .context("moshi::mimi::load")?;
    eprintln!("[moshi-rs] worker: mimi loaded in {:?}", t1.elapsed());

    let t2 = std::time::Instant::now();
    let lm_model = lm::load_lm_model(engine.lm_cfg.clone(), &engine.paths.lm_weights, dtype, &device)
        .context("moshi::lm::load_lm_model")?;
    eprintln!("[moshi-rs] worker: lm loaded in {:?}", t2.elapsed());

    let audio_lp = LogitsProcessor::from_sampling(
        sess_cfg.seed,
        Sampling::TopK { k: 250, temperature: 0.8 },
    );
    let text_lp = LogitsProcessor::from_sampling(
        sess_cfg.seed,
        Sampling::TopK { k: 250, temperature: 0.8 },
    );

    let max_steps = sess_cfg.max_steps.max(1) as usize;

    let mut state = lm_generate_multistream::State::new(
        lm_model,
        max_steps + 20,
        audio_lp,
        text_lp,
        None,
        None,
        None,
        lm_generate_multistream::Config {
            acoustic_delay: 2,
            audio_vocab_size: engine.lm_cfg.audio_vocab_size,
            generated_audio_codebooks: engine.generated_audio_codebooks,
            input_audio_codebooks: engine.lm_cfg.audio_codebooks - engine.generated_audio_codebooks,
            text_start_token: engine.lm_cfg.text_out_vocab_size as u32,
            text_eop_token: 0,
            text_pad_token: 3,
        },
    );

    let mut prev_text_token = state.config().text_start_token;
    eprintln!("[moshi-rs] worker: state ready, entering streaming loop");

    loop {
        if cancelled.load(Ordering::Acquire) {
            event_tx
                .send(RustEvent::SessionCompleted {
                    status: Status::Preempted,
                    message: "cancelled by caller".to_owned(),
                })
                .ok();
            return Ok(());
        }

        let pcm = match audio_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(v) => v,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                event_tx
                    .send(RustEvent::SessionCompleted {
                        status: Status::Ok,
                        message: "input channel closed".to_owned(),
                    })
                    .ok();
                return Ok(());
            }
        };
        if pcm.len() != FRAME_SAMPLES {
            event_tx
                .send(RustEvent::Error {
                    status: Status::InvalidInput,
                    message: format!(
                        "session_worker: expected {FRAME_SAMPLES} samples, got {}",
                        pcm.len()
                    ),
                })
                .ok();
            continue;
        }

        // Mimi encode: 80 ms PCM → audio tokens.
        let frame_started = std::time::Instant::now();
        let pcm_tensor = Tensor::from_vec(pcm, (1, 1, FRAME_SAMPLES), &device)
            .context("Tensor::from_vec input pcm")?;
        let in_codes = mimi_model
            .encode_step(&pcm_tensor.into(), &().into())
            .context("mimi.encode_step")?;
        let in_codes_t = match in_codes.as_option() {
            Some(t) => t.clone(),
            None => {
                eprintln!("[moshi-rs] worker: encode_step returned None (frame ignored, encoder warming)");
                continue;
            }
        };
        eprintln!(
            "[moshi-rs] worker: encode_step returned codes, encode took {:?}",
            frame_started.elapsed()
        );

        // The encoder may emit multiple steps per call; iterate.
        let (_b, _codebooks, steps) = in_codes_t.dims3().context("in_codes dims3")?;
        for step in 0..steps {
            if cancelled.load(Ordering::Acquire) {
                event_tx
                    .send(RustEvent::SessionCompleted {
                        status: Status::Preempted,
                        message: "cancelled by caller".to_owned(),
                    })
                    .ok();
                return Ok(());
            }
            let codes = in_codes_t
                .i((.., .., step..step + 1))
                .context("index in_codes step")?;
            let codes = codes.i((0, .., 0)).context("squeeze in_codes")?;
            let codes = codes.to_vec1::<u32>().context("to_vec1 in_codes")?;

            let lm_started = std::time::Instant::now();
            prev_text_token = state
                .step_(Some(prev_text_token), &codes, None, None, None)
                .context("state.step_")?;
            eprintln!(
                "[moshi-rs] worker: state.step_ took {:?} (text_tok={})",
                lm_started.elapsed(),
                prev_text_token
            );

            // Emit transcript chunk if we got a real text token.
            if prev_text_token != state.config().text_pad_token
                && prev_text_token != state.config().text_start_token
                && prev_text_token != state.config().text_eop_token
            {
                if let Ok(piece) = engine.text_tokenizer.decode_piece_ids(&[prev_text_token]) {
                    if !piece.is_empty() {
                        event_tx
                            .send(RustEvent::TranscriptChunk { text: piece, is_final: false })
                            .ok();
                    }
                }
            }

            // Decode audio tokens for this step.
            if let Some(audio_tokens) = state.last_audio_tokens() {
                let audio_tokens =
                    Tensor::new(&audio_tokens[..engine.generated_audio_codebooks], &device)
                        .context("Tensor::new audio_tokens")?
                        .reshape((1, 1, ()))
                        .context("reshape audio_tokens")?
                        .t()
                        .context("transpose audio_tokens")?;
                let out_pcm_t = mimi_model
                    .decode_step(&audio_tokens.into(), &().into())
                    .context("mimi.decode_step")?;
                if let Some(t) = out_pcm_t.as_option() {
                    let pcm_out = t
                        .i((0, 0))
                        .context("index decode_step output")?
                        .to_vec1::<f32>()
                        .context("to_vec1 pcm_out")?;
                    event_tx.send(RustEvent::AudioChunk(pcm_out)).ok();
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// C ABI — engine
// -----------------------------------------------------------------------

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
    let cfg = unsafe { &*cfg };
    match check_artifacts(cfg) {
        Ok(_) => Status::Ok,
        Err(msg) => {
            unsafe { write_err(err_buf, err_buflen, &msg) };
            Status::ArtifactMissing
        }
    }
}

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
    let cfg = unsafe { &*cfg };
    let paths = match check_artifacts(cfg) {
        Ok(p) => p,
        Err(msg) => {
            unsafe { write_err(err_buf, err_buflen, &msg) };
            return Status::ArtifactMissing;
        }
    };
    let lm_cfg = match read_lm_config(paths.lm_config.as_ref()) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("{e:#}");
            unsafe { write_err(err_buf, err_buflen, &msg) };
            return Status::LoadFailed;
        }
    };
    let generated_audio_codebooks = lm_cfg.depformer.as_ref().map_or(8, |v| v.num_slices);
    let text_tokenizer = match sentencepiece::SentencePieceProcessor::open(&paths.text_tokenizer) {
        Ok(t) => Arc::new(t),
        Err(e) => {
            let msg = format!("sentencepiece open {}: {e:#}", paths.text_tokenizer.display());
            unsafe { write_err(err_buf, err_buflen, &msg) };
            return Status::LoadFailed;
        }
    };

    let inner = EngineInner {
        paths,
        use_metal: cfg.use_metal != 0,
        text_tokenizer,
        lm_cfg,
        generated_audio_codebooks,
        session_lock: Mutex::new(()),
    };
    let engine = Engine { inner: Arc::new(inner) };
    let boxed = Box::new(engine);
    unsafe { *out_engine = Box::into_raw(boxed) };
    Status::Ok
}

#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_engine_close(engine: *mut Engine) {
    if engine.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(engine) });
}

// -----------------------------------------------------------------------
// C ABI — session
// -----------------------------------------------------------------------

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
    let engine = unsafe { &*engine };
    let cfg = unsafe { &*cfg };

    // Single-session enforcement.
    let session_lock_static = unsafe {
        std::mem::transmute::<&Mutex<()>, &'static Mutex<()>>(&engine.inner.session_lock)
    };
    let guard = match session_lock_static.try_lock() {
        Ok(g) => g,
        Err(_) => {
            unsafe { write_err(err_buf, err_buflen, "session_open: another session is already active (Slice 2C is single-session)") };
            return Status::InitFailed;
        }
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
    let (event_tx, event_rx) = mpsc::channel::<RustEvent>();

    let cancel_for_worker = Arc::clone(&cancelled);
    let engine_for_worker = Arc::clone(&engine.inner);
    let sess_cfg = SessionConfig { seed: cfg.seed, max_steps: cfg.max_steps };
    let worker = std::thread::Builder::new()
        .name("octomil-moshi-rs-session".into())
        .spawn(move || {
            session_worker(engine_for_worker, sess_cfg, audio_rx, event_tx, cancel_for_worker);
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
        _engine_session_lock: Some(guard),
    };
    let boxed = Box::new(session);
    unsafe { *out_session = Box::into_raw(boxed) };
    Status::Ok
}

#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_send_audio(
    session: *mut Session,
    pcm: *const f32,
    n_samples: usize,
) -> Status {
    if session.is_null() || pcm.is_null() {
        return Status::InvalidInput;
    }
    if n_samples != FRAME_SAMPLES {
        return Status::InvalidInput;
    }
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

#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_pop_event(
    session: *mut Session,
    out_event: *mut CEvent,
) -> Status {
    if session.is_null() || out_event.is_null() {
        return Status::InvalidInput;
    }
    let session = unsafe { &*session };
    let mut last_event = session.last_event.lock().expect("last_event poisoned");
    let mut last_message = session.last_message_cstr.lock().expect("last_message_cstr poisoned");
    let mut last_transcript = session.last_transcript_cstr.lock().expect("last_transcript_cstr poisoned");

    let next = {
        let rx = session.event_rx.lock().expect("event_rx poisoned");
        match rx.try_recv() {
            Ok(ev) => ev,
            Err(mpsc::TryRecvError::Empty) | Err(mpsc::TryRecvError::Disconnected) => {
                unsafe { *out_event = CEvent::empty() };
                return Status::Again;
            }
        }
    };

    let mut c_event = CEvent::empty();
    match &next {
        RustEvent::AudioChunk(pcm) => {
            c_event.event_type = EventType::AudioChunk;
            c_event.audio_pcm = pcm.as_ptr();
            c_event.audio_n_samples = pcm.len();
        }
        RustEvent::TranscriptChunk { text, is_final } => {
            c_event.event_type = EventType::TranscriptChunk;
            *last_transcript = Some(CString::new(text.as_bytes()).unwrap_or_default());
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

#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_cancel(session: *mut Session) -> Status {
    if session.is_null() {
        return Status::InvalidInput;
    }
    let session = unsafe { &*session };
    session.cancelled.store(true, Ordering::Release);
    Status::Ok
}

#[no_mangle]
pub unsafe extern "C" fn octomil_moshi_rs_session_close(session: *mut Session) {
    if session.is_null() {
        return;
    }
    let mut session = unsafe { Box::from_raw(session) };
    session.audio_tx.take();
    session.cancelled.store(true, Ordering::Release);
    if let Some(worker) = session.worker.take() {
        let _ = worker.join();
    }
    // Engine session lock released on drop.
}

#[no_mangle]
pub extern "C" fn octomil_moshi_rs_version() -> *const c_char {
    concat!("octomil-moshi-rs/", env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nul_terminated_static() {
        let p = octomil_moshi_rs_version();
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) };
        assert!(s.to_str().unwrap().starts_with("octomil-moshi-rs/"));
    }
}
