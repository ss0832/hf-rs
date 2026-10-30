// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Installed command line, sharing the native calculation API.
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|s| s == "--help" || s == "-h") {
        print!("{}", hf_rs::cli::HELP);
        return;
    }
    if args == ["--version"] {
        println!("hf-rs {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    match hf_rs::cli::execute(&args) {
        Ok(value) => {
            println!("{value}");
            if value.get("passed").and_then(|v| v.as_bool()) == Some(false) {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
