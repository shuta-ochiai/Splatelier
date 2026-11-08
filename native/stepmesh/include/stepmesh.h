// SPDX-License-Identifier: Apache-2.0
// C interface of the stepmesh static library (native/stepmesh).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Converts the STEP file at `input_path` to a USDZ file at `output_path`.
/// Returns NULL on success, otherwise an error message to free with `stepmesh_free_string`.
char *stepmesh_convert_to_usdz(const char *input_path, const char *output_path);

/// Frees a string returned by `stepmesh_convert_to_usdz`.
void stepmesh_free_string(char *message);

#ifdef __cplusplus
}
#endif
