// SPDX-License-Identifier: Apache-2.0
//! STEP to USDZ conversion for the Splatelier viewer.
//!
//! cadmpeg decodes the STEP file (B-rep, product structure, placements, colours) and its USD
//! writer emits a package whose display meshes ModelIO can read. The C entry points are what the
//! Swift side links against; see `include/stepmesh.h`.

use std::ffi::{c_char, CStr, CString};
use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cadmpeg_codec_step::StepCodec;
use cadmpeg_codec_usd::UsdCodec;
use cadmpeg_core::decode::DecodeDepth;
use cadmpeg_ir::codec::write::{target::TargetRequest, EncodeInput, Encoder};
use cadmpeg_ir::codec::{Codec, DecodeOptions};

/// The USDZ package of a STEP file's bytes.
///
/// # Errors
///
/// Returns a readable reason when the file is not STEP, cannot be decoded, or has nothing the
/// USD writer can emit.
pub fn step_to_usdz(step: &[u8]) -> Result<Vec<u8>, String> {
    let options = DecodeOptions { depth: DecodeDepth::Full, ..DecodeOptions::default() };
    let decoded = StepCodec::default()
        .decode(&mut Cursor::new(step), &options)
        .map_err(|error| format!("STEP decode failed: {error}"))?;
    let plan = UsdCodec
        .plan(EncodeInput::new(decoded.ir(), None), TargetRequest::Explicit("usd:usdz"))
        .map_err(|error| format!("USD export failed: {error}"))?;
    let mut bytes = Vec::new();
    plan.write_to(&mut bytes).map_err(|error| format!("USD write failed: {error}"))?;
    Ok(bytes)
}

fn convert(input: &CStr, output: &CStr) -> Result<(), String> {
    let input = input.to_str().map_err(|_| "input path is not UTF-8".to_owned())?;
    let output = output.to_str().map_err(|_| "output path is not UTF-8".to_owned())?;
    let step = std::fs::read(input).map_err(|error| format!("cannot read {input}: {error}"))?;
    let usdz = step_to_usdz(&step)?;
    std::fs::write(output, usdz).map_err(|error| format!("cannot write {output}: {error}"))
}

/// Converts the STEP file at `input_path` to a USDZ file at `output_path`.
///
/// Returns null on success, otherwise an error message the caller frees with
/// [`stepmesh_free_string`]. A panic inside cadmpeg is caught and returned as an error.
///
/// # Safety
///
/// Both arguments must be valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn stepmesh_convert_to_usdz(
    input_path: *const c_char,
    output_path: *const c_char,
) -> *mut c_char {
    if input_path.is_null() || output_path.is_null() {
        return error_string("null path".to_owned());
    }
    let (input, output) = (CStr::from_ptr(input_path), CStr::from_ptr(output_path));
    match catch_unwind(AssertUnwindSafe(|| convert(input, output))) {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(message)) => error_string(message),
        Err(_) => error_string("STEP conversion crashed (cadmpeg panicked)".to_owned()),
    }
}

/// Frees a string returned by [`stepmesh_convert_to_usdz`].
///
/// # Safety
///
/// `message` must be null or a pointer returned by this library, freed once.
#[no_mangle]
pub unsafe extern "C" fn stepmesh_free_string(message: *mut c_char) {
    if !message.is_null() {
        drop(CString::from_raw(message));
    }
}

fn error_string(message: String) -> *mut c_char {
    CString::new(message.replace('\0', " ")).map_or(std::ptr::null_mut(), CString::into_raw)
}
