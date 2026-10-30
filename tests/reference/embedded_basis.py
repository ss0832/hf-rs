"""Read the exact embedded G94 data instead of differently rounded program tables."""

import hashlib
import re
from pathlib import Path

from pyscf.gto.basis import parse_gaussian

SOURCE = Path(__file__).resolve().parents[2] / "src/basis_data.rs"


def embedded_basis(constant, symbols):
    text = SOURCE.read_text(encoding="utf-8")
    gbs = re.search(rf'pub const {constant}: &str = r#"(.*?)"#;', text, re.S).group(1)
    result = {}
    for symbol in symbols:
        segment = re.search(rf"(?m)^{symbol}\s+0\s*\n(.*?)\*\*\*\*", gbs, re.S).group(1)
        result[symbol] = parse_gaussian.parse(segment, optimize=False)
    return result


def basis_hash():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()
