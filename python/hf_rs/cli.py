# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Python CLI using the same native run function as Rust and ASE."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import find_atomic_number, method_catalog, oracle_compare, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hf-rs")
    parser.add_argument("--version", action="version", version="hf-rs " + __version__)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--oracle-compare", action="store_true")
    parser.add_argument("--method", default="hf")
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--geometry", default="H 0 0 -0.7; H 0 0 0.7")
    parser.add_argument("--unit", choices=("bohr", "angstrom"))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ecp")
    parser.add_argument("--spherical", action="store_true")
    parser.add_argument("--cartesian", action="store_true")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--reference", choices=("auto", "rhf", "uhf"), default="auto")
    parser.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--property",
        action="append",
        choices=("energy", "gradient", "hessian", "observables", "intermediates"),
    )
    args = parser.parse_args(argv)
    try:
        if args.cartesian and args.spherical:
            raise ValueError("choose only one of --cartesian and --spherical")
        if args.oracle_compare or args.list_methods:
            actual = sys.argv[1:] if argv is None else argv
            if len(actual) != 1:
                raise ValueError(
                    "catalog and oracle flags take no calculation arguments"
                )
            result = oracle_compare() if args.oracle_compare else method_catalog()
            print(json.dumps(result, allow_nan=False))
            if isinstance(result, dict) and result.get("passed") is False:
                parser.exit(1)
            return
        if args.xyz:
            lines = args.xyz.read_text(encoding="utf-8").splitlines()
            count = int(lines[0].strip())
            lines = lines[2:]
            if len(lines) != count:
                raise ValueError("XYZ atom count mismatch")
        else:
            lines = args.geometry.split(";")
        numbers, positions = [], []
        for line in lines:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("geometry rows require SYMBOL X Y Z")
            number = find_atomic_number(fields[0])
            if number is None:
                raise ValueError("unknown element: " + fields[0])
            numbers.append(number)
            positions.append([float(x) for x in fields[1:]])
        options = {}
        for item in args.opt:
            key, value = item.split("=", 1)
            if not key or key in options:
                raise ValueError("empty or duplicate option: " + key)
            try:
                options[key] = json.loads(value)
            except json.JSONDecodeError:
                options[key] = value
        result = run(
            args.method,
            numbers,
            positions,
            unit=args.unit or ("angstrom" if args.xyz else "bohr"),
            basis=args.basis,
            ecp=args.ecp,
            cartesian=not args.spherical,
            charge=args.charge,
            multiplicity=args.multiplicity,
            reference=args.reference,
            options=options,
            properties=args.property,
        )
        print(json.dumps(result, allow_nan=False))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
