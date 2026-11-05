// SPDX-License-Identifier: Apache-2.0
//! Converts the furniture fixtures written by `scripts/make_step_fixtures.py`.

use std::path::PathBuf;

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../Tests/SplatViewerKitTests/Fixtures/STEP")
        .join(format!("{name}.step"));
    std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn every_fixture_converts_to_a_usdz_package() {
    for name in ["table", "chair", "side_table", "bookshelf"] {
        let usdz = stepmesh::step_to_usdz(&fixture(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
        // A USDZ package is a stored ZIP archive.
        assert_eq!(&usdz[..4], b"PK\x03\x04", "{name}");
    }
}

#[test]
fn a_file_that_is_not_step_is_refused() {
    assert!(stepmesh::step_to_usdz(b"solid cube\nendsolid cube\n").is_err());
}
