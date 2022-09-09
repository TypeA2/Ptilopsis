#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="Run tests for the Ptilopsis compiler backend")

def make_schema(p: Path) -> list[Path]:
    rel = lambda x: (p / x).resolve()
    return [
        rel("./basic/1.in"),
        rel("./basic/2.in"),
        rel("./basic/3.in"),
        rel("./basic/4.in"),
        rel("./basic/5.in"),
        rel("./basic/6.in"),
        rel("./basic/7.in"),
        rel("./funclen/1.in"),
        rel("./funclen/2.in"),
        rel("./funclen/3.in"),
        rel("./funclen/4.in"),
        rel("./funclen/5.in"),
        rel("./funclen/6.in"),
        rel("./funclen/7.in"),
        rel("./funclen/8.in"),
        rel("./shape/1.in"),
        rel("./shape/2.in"),
        rel("./shape/3.in"),
        rel("./shape/4.in"),
        rel("./shape/5.in"),
        rel("./shape/6.in"),
    ]

def executable(p) -> Path:
    p = Path(p).resolve()
    if not p.exists():
        parser.error(f"File {p} does not exist.")
    elif not os.access(p, os.X_OK):
        parser.error(f"File {p} is not executable.")
    else:
        return p

def dir(p) -> Path:
    p = Path(p).resolve()
    if not p.exists():
        parser.error(f"Path {p} does not exist.")
    elif not p.is_dir():
        parser.error(f"Path {p} is not a directory")
    else:
        schema = make_schema(p)
        for x in schema:
            if not (x.exists() and x.is_file()):
                parser.error(f"Testdir invalid: {x.relative_to(Path('.').resolve())} is not a file or does not exist")

        return p

parser.add_argument("executable", type=executable, help="Ptilopsis executable to use")
parser.add_argument("testdir", type=dir, help="Directory containing test files")

args, unknown = parser.parse_known_args()

ptilopsis: Path = args.executable
testdir: Path = args.testdir
schema: list[Path] = make_schema(testdir)
runs = 15

print(f"{runs} run(s)", file=sys.stderr)

cwd = Path(".").resolve()

digits = len(str(runs))

results: dict[Path, list[dict[str, int]]] = {}

t = 0

for file in schema:
    results[file] = []

    for i in range(runs):
        print(f"> {file.relative_to(testdir)}  ->  {str(i + 1).ljust(digits, ' ')} / {runs}...", file=sys.stderr, end="")
        argv: list[str | Path] = [
            ptilopsis,
            "-p",
            "-o", os.devnull,
            file,
            *unknown
        ]

        start = time.time()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        stdout, _ = proc.communicate()

        elapsed = time.time() - start
        print(f" {elapsed} seconds", file=sys.stderr)

        rows: list[str] = stdout.split("\n")
        header: list[str] = rows[0].split(",")
        counts: list[int] = list(map(int, rows[1].split(",")))

        print(f"{file.relative_to(testdir)},{counts}")
        sys.stdout.flush()
