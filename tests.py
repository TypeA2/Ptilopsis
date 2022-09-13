#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
from typing import TextIO

parser = argparse.ArgumentParser(description="Run tests for the Ptilopsis compiler backend")

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
        return p

parser.add_argument("executable", type=executable, help="Ptilopsis executable to use")
parser.add_argument("testdir", type=dir, help="Directory containing test files")
parser.add_argument("outfile", type=argparse.FileType("w", encoding="utf-8"), help="File to write output to")

args = parser.parse_args()

ptilopsis: Path = args.executable
testdir: Path = args.testdir
out: TextIO = args.outfile

threads_to_test = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224]
alt_block_to_test = [True, False]
lock_types_to_test = [0, 1]

with out:
    for thread_count in threads_to_test:
        for alt_block in alt_block_to_test:
            for lock_type in lock_types_to_test:
                argv: list[Path | str] = [
                    Path("./test.py").resolve(),
                    ptilopsis,
                    testdir,
                    "-t", str(thread_count),
                    "-m", str(lock_type)
                ]

                if alt_block:
                    argv.append("-b")

                with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1) as proc:
                    print(f"threads={thread_count}, altblock={alt_block}, locktype={lock_type}")

                    for line in iter(proc.stdout.readline, ""):
                        line = f"{thread_count},{int(alt_block)},{lock_type},{line}"
                        out.write(line)
                        out.flush()
                        print(f"  > {line}", end="")
print("Done")
