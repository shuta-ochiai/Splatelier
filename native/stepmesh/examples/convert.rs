// SPDX-License-Identifier: Apache-2.0
//! `cargo run --example convert -- in.step out.usdz`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = args.as_slice() else {
        eprintln!("usage: convert <in.step> <out.usdz>");
        std::process::exit(2);
    };
    let step = std::fs::read(input).expect("read the STEP file");
    match stepmesh::step_to_usdz(&step) {
        Ok(usdz) => std::fs::write(output, usdz).expect("write the USDZ file"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
