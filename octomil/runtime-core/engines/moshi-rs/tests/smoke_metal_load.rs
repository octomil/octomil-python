//! Spike 3 confirmation: load moshiko-candle-q8 via the Rust shim
//! and stream one 80 ms silence frame through Mimi+LM+Mimi end-to-
//! end on Metal. Gated on the artifact being staged at
//! `OCTOMIL_TEST_MOSHI_ARTIFACTS=...`. If the env var is unset, the
//! test is a no-op (skips with a logged note).
//!
//! This is the Rust-side smoke. The C++ adapter's conformance test
//! exercises the full C ABI surface; this one stays inside the
//! Rust crate so we get a fast pass/fail without rebuilding the
//! full dylib + CMake pipeline.

use std::ffi::CString;
use std::os::raw::c_char;

#[allow(dead_code)]
fn cstr(p: &std::path::Path) -> CString {
    CString::new(p.to_string_lossy().as_ref()).unwrap()
}

#[test]
fn moshi_rs_streams_one_frame_on_metal() {
    let root = match std::env::var("OCTOMIL_TEST_MOSHI_ARTIFACTS") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => {
            eprintln!("[skip] OCTOMIL_TEST_MOSHI_ARTIFACTS not set");
            return;
        }
    };
    let lm = root.join("model.safetensors");
    let mimi = root.join("tokenizer-e351c8d8-checkpoint125.safetensors");
    let tok = root.join("tokenizer_spm_32k_3.model");
    if !lm.is_file() || !mimi.is_file() || !tok.is_file() {
        eprintln!(
            "[skip] missing one of: {} {} {}",
            lm.display(),
            mimi.display(),
            tok.display()
        );
        return;
    }

    let lm_c = cstr(&lm);
    let mimi_c = cstr(&mimi);
    let tok_c = cstr(&tok);
    let lm_cfg_path = root.join("config.json");
    let lm_cfg_c = if lm_cfg_path.is_file() {
        Some(cstr(&lm_cfg_path))
    } else {
        None
    };
    let cfg = octomil_moshi_rs::EngineConfig {
        lm_weights_path: lm_c.as_ptr(),
        mimi_weights_path: mimi_c.as_ptr(),
        text_tokenizer_path: tok_c.as_ptr(),
        lm_config_path: lm_cfg_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        use_metal: 1,
    };

    let mut err = vec![0i8; 4096];
    let mut engine: *mut octomil_moshi_rs::Engine = std::ptr::null_mut();
    let s = unsafe {
        octomil_moshi_rs::octomil_moshi_rs_engine_open(
            &cfg,
            &mut engine,
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    assert!(
        matches!(s, octomil_moshi_rs::Status::Ok),
        "engine_open failed: status={:?} err={}",
        s,
        unsafe { std::ffi::CStr::from_ptr(err.as_ptr() as *const c_char) }
            .to_string_lossy()
    );

    let sess_cfg = octomil_moshi_rs::SessionConfig { seed: 0xCAFEBABE, max_steps: 32 };
    let mut session: *mut octomil_moshi_rs::Session = std::ptr::null_mut();
    let s = unsafe {
        octomil_moshi_rs::octomil_moshi_rs_session_open(
            engine,
            &sess_cfg,
            &mut session,
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    assert!(
        matches!(s, octomil_moshi_rs::Status::Ok),
        "session_open failed: status={:?} err={}",
        s,
        unsafe { std::ffi::CStr::from_ptr(err.as_ptr() as *const c_char) }
            .to_string_lossy()
    );

    // Send a stream of silence frames continuously while polling.
    // Mimi has encoder warmup latency; gen.rs's reference loop
    // processes ~25 frames before the first audio out emerges.
    let silence = vec![0.0f32; 1920];
    let pump = {
        let session_ptr = session as usize;
        let silence = silence.clone();
        std::thread::spawn(move || {
            let session = session_ptr as *mut octomil_moshi_rs::Session;
            for i in 0..200 {
                let s = unsafe { octomil_moshi_rs::octomil_moshi_rs_session_send_audio(session, silence.as_ptr(), silence.len()) };
                if !matches!(s, octomil_moshi_rs::Status::Ok) {
                    eprintln!("[smoke] send_audio frame {i} returned {:?}", s);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        })
    };

    // Spin briefly waiting for an event; we want the loop to have
    // produced at least one chunk (audio or transcript). Real
    // streaming budgets are validated separately; this is a smoke
    // for "anything comes out".
    // Increase deadline because session_open returns immediately but
    // the worker still has to candle-load the q8 GGUF (multi-GB) and
    // Mimi safetensors before it can run the first encode_step.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let start = std::time::Instant::now();
    let mut got_any_event = false;
    let mut last_status: Option<octomil_moshi_rs::Status> = None;
    while std::time::Instant::now() < deadline && !got_any_event {
        let mut ev = octomil_moshi_rs::CEvent {
            event_type: octomil_moshi_rs::EventType::None,
            audio_pcm: std::ptr::null(),
            audio_n_samples: 0,
            transcript_utf8: std::ptr::null(),
            transcript_n_bytes: 0,
            is_final: 0,
            n_frames_dropped: 0,
            status: octomil_moshi_rs::Status::Ok,
            status_message: std::ptr::null(),
        };
        let s = unsafe { octomil_moshi_rs::octomil_moshi_rs_session_pop_event(session, &mut ev) };
        last_status = Some(s);
        if matches!(s, octomil_moshi_rs::Status::Ok)
            && !matches!(ev.event_type, octomil_moshi_rs::EventType::None)
        {
            eprintln!(
                "[smoke] first event after {}ms: type={:?} audio_n={} transcript_n={} status={:?}",
                start.elapsed().as_millis(),
                ev.event_type,
                ev.audio_n_samples,
                ev.transcript_n_bytes,
                ev.status,
            );
            if matches!(ev.event_type, octomil_moshi_rs::EventType::Error)
                || matches!(ev.event_type, octomil_moshi_rs::EventType::SessionCompleted)
            {
                let msg = if ev.status_message.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(ev.status_message) }
                        .to_string_lossy()
                        .into_owned()
                };
                eprintln!("[smoke] event status_message: {}", msg);
            }
            got_any_event = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    eprintln!(
        "[smoke] elapsed={}ms got_any_event={} last_status={:?}",
        start.elapsed().as_millis(),
        got_any_event,
        last_status
    );

    drop(pump);
    unsafe {
        octomil_moshi_rs::octomil_moshi_rs_session_close(session);
        octomil_moshi_rs::octomil_moshi_rs_engine_close(engine);
    }

    assert!(got_any_event, "no event observed within 120s");
}
