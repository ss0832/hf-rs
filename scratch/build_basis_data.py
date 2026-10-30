import argparse
import os

parser = argparse.ArgumentParser(
    description="Assemble externally supplied basis fragments."
)
parser.add_argument(
    "fragments_directory", help="Directory containing the basis fragments"
)
parser.add_argument(
    "output", help="Explicit destination for the generated candidate Rust file"
)
args = parser.parse_args()
steps_dir = args.fragments_directory

files = {
    "BASIS_3_21G_GBS": os.path.join(steps_dir, "13337", "content.md"),
    "BASIS_6_31G_STAR_STAR_GBS": os.path.join(steps_dir, "13339", "content.md"),
    "BASIS_DEF2_SVP_GBS": os.path.join(steps_dir, "13341", "content.md"),
    "BASIS_DEF2_TZVP_GBS": os.path.join(steps_dir, "13343", "content.md"),
    "BASIS_CC_PVDZ_GBS": os.path.join(steps_dir, "13345", "content.md"),
    "BASIS_CC_PVTZ_GBS": os.path.join(steps_dir, "13347", "content.md"),
    "BASIS_AUG_CC_PVDZ_GBS": os.path.join(steps_dir, "13349", "content.md"),
}

out_path = args.output

header = """// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Built-in standard basis sets sourced from BasisSetExchange (BSE).
//! Includes standard Pople family, Karlsruhe (Ahlrichs def2) family, and Dunning correlation-consistent (cc-pVXZ) family.

"""

with open(out_path, "w", encoding="utf-8") as out:
    out.write(header)
    for const_name, path in files.items():
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Find where actual content begins (line starting with ! Basis Set Exchange or element)
        start_idx = 0
        for i, line in enumerate(lines):
            if (
                line.startswith("!")
                or line.strip() == "****"
                or (len(line.split()) == 2 and line.split()[1] == "0")
            ):
                start_idx = i
                break
        clean_content = "".join(lines[start_idx:]).rstrip()
        out.write(
            f"/// {const_name} standard basis set in Gaussian 94 format from BasisSetExchange\n"
        )
        out.write(f'pub const {const_name}: &str = r#"\n{clean_content}\n"#;\n\n')

print(f"Wrote {out_path} successfully!")
