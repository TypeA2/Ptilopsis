#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import TextIO

parser = argparse.ArgumentParser(description="Run tests for the Ptilopsis compiler backend")

def make_schema(p: Path) -> list[Path]:
    rel = lambda x: (p / x).resolve()
    return [
        rel("./basic/1.in"),
        rel("./basic/2.in"),
        rel("./basic/3.in"),
        rel("./basic/4.in"),
        rel("./basic/5.in"),
        rel("./funclen/1.in"),
        rel("./funclen/2.in"),
        rel("./funclen/3.in"),
        rel("./funclen/4.in"),
        rel("./funclen/5.in"),
        rel("./funclen/6.in"),
        rel("./funclen/7.in"),
        rel("./funclen/8.in")
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
parser.add_argument("outfile", type=argparse.FileType("w", encoding="utf-8"), help="File to write output to")

args, unknown = parser.parse_known_args()

ptilopsis: Path = args.executable
testdir: Path = args.testdir
out: TextIO = args.outfile
schema: list[Path] = make_schema(testdir)
runs = 15

print(f"{runs} run(s)", file=sys.stderr)

with out:
    for file in schema:
        procs: list[subprocess.Popen] = []
        print(f" > {file.relative_to(testdir)}...")
        for i in range(runs):
            argv: list[str | Path] = [
                ptilopsis,
                "-p", "-s",
                "-o", os.devnull,
                file
            ]

            procs.append(subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8"))

        for proc in procs:
            stdout, _ = proc.communicate()

            rows: list[str] = stdout.split("\n")
            header: list[str] = rows[0].split(",")
            counts: list[int] = list(map(int, rows[1].split(",")))

            out.write(f"{file.relative_to(testdir)},{counts}\n")
            out.flush()
