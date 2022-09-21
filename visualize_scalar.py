#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO, TypeVar, cast
import argparse
import numpy
import matplotlib.axes
import matplotlib.pyplot as plt
import pylab

parser = argparse.ArgumentParser(description="Visualizations for Ptilopsis")

def outdir_check(v: str) -> Path:
    p = Path(v).resolve()

    if not p.exists():
        parser.error(f"Path \"{v}\" does note exist")
    elif not p.is_dir():
        parser.error(f"Path \"{v}\" is not a directory!")
    
    return p

parser.add_argument("infile_scalar", type=argparse.FileType("r", encoding="utf-8"), help="Scalar results .csv")
parser.add_argument("infile_avx", type=argparse.FileType("r", encoding="utf-8"), help="AVX results .csv")
parser.add_argument("outdir", type=outdir_check, help="Output image directory")

@dataclass
class TestResult:
    infile: str

    preprocess: int
    isn_cnt: int
    isn_gen: int
    optimize: int
    regalloc: int
    fix_jumps: int
    postprocess: int

    @classmethod
    def parse(cls, line: str) -> TestResult:
        infile, values = line.split(",", 1)
        
        durations = list(map(int, values.lstrip("[").rstrip("]").split(", ")[:-1]))

        return TestResult(
            infile,
            durations[0], durations[1], durations[2], durations[3], durations[4], durations[5], durations[6]
        )

    def __str__(self) -> str:
        post: str = f"{self.preprocess},{self.isn_cnt},{self.isn_gen},{self.optimize},{self.regalloc},{self.fix_jumps},{self.postprocess}"
        return f"{self.infile},{post}"

    def total_time(self) -> int:
        return self.preprocess + self.isn_cnt + self.isn_gen + self.optimize + self.regalloc + self.fix_jumps + self.postprocess

    def get(self, attr: str) -> int:
        return self.__getattribute__(attr)

SelectReturn = TypeVar("SelectReturn")

class TestResultArray(list[TestResult]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, val: TestResult | str) -> None:
        if type(val) == str:
            val = TestResult.parse(cast(str, val))

        super().append(cast(TestResult, val))

    def average(self) -> TestResult | None:
        if len(self) == 0:
            return None

        res = TestResult(
            self[0].infile, 0, 0, 0, 0, 0, 0, 0
        )

        for val in self:
            res.preprocess += val.preprocess
            res.isn_cnt += val.isn_cnt
            res.isn_gen += val.isn_gen
            res.optimize += val.optimize
            res.regalloc += val.regalloc
            res.fix_jumps += val.fix_jumps
            res.postprocess += val.postprocess

        res.preprocess = round(res.preprocess / len(self))
        res.isn_cnt = round(res.isn_cnt / len(self))
        res.isn_gen = round(res.isn_gen / len(self))
        res.optimize = round(res.optimize / len(self))
        res.regalloc = round(res.regalloc / len(self))
        res.fix_jumps = round(res.fix_jumps / len(self))
        res.postprocess = round(res.postprocess / len(self))

        return res

    def filter(self, predicate: Callable[[TestResult], bool]) -> TestResultArray:
        return TestResultArray(filter(predicate, self))

    def select(self, predicate: str | Callable[[TestResult], SelectReturn]) -> list[SelectReturn]:
        if type(predicate) == str:
            return [ v.__getattribute__(cast(str, predicate)) for v in self ]
        else:
            return [ cast(Callable, predicate)(v) for v in self ]
            
args = parser.parse_args()

infile_scalar: TextIO = args.infile_scalar
infile_avx: TextIO = args.infile_avx
outdir: Path = args.outdir

schema = [
    "basic/1.in",
    "basic/2.in",
    "basic/3.in",
    "basic/4.in",
    "basic/5.in",
    "basic/6.in",
    "basic/7.in",
    "funclen/1.in",
    "funclen/2.in",
    "funclen/3.in",
    "funclen/4.in",
    "funclen/5.in",
    "funclen/6.in",
    "funclen/7.in",
    "funclen/8.in",
    "shape/1.in",
    "shape/2.in",
    "shape/3.in",
    "shape/4.in",
    "shape/5.in",
    "shape/6.in",
]

basic_size = [
    5120, 10240, 102400, 512000, 1048576, 10485760, 52428800
]

funclen_count = [
    25118, 12495, 6181, 2513, 1539, 497, 251, 100
]

shape_depth = [
    9, 19, 34, 36, 42, 49
]

basic_scaling = [ 1, 2, 20, 100, 200, 2000, 10000 ]

for file in schema:
    try:
        Path(outdir / file).parent.mkdir(parents=True)
    except FileExistsError:
        pass

values_scalar = TestResultArray()
for line in infile_scalar:
    values_scalar.append(line.rstrip("\n"))

values_avx = TestResultArray()
values_avx_st = TestResultArray()
for line in infile_avx:
    threads, alt_block, lock_type, infile, values = line.split(",", 4)

    if infile in schema and alt_block == "1" and lock_type == "1":
        if threads == "16":
            values_avx.append(f"{infile},{values}")
        elif threads == "1":
            values_avx_st.append(f"{infile},{values}")

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Computer Modern"],
    "text.usetex": True
})

schema_basic = list(filter(lambda t: t.startswith("basic"), schema))
schema_funclen = list(filter(lambda t: t.startswith("funclen"), schema))
schema_shape = list(filter(lambda t: t.startswith("shape"), schema))

plot_basic_scalar: list[float] = []
plot_basic_avx: list[float] = []
plot_basic_avx_st: list[float] = []
for file in schema_basic:
    if scalar_avg := values_scalar.filter(lambda t: t.infile == file).average():
        plot_basic_scalar.append(scalar_avg.total_time() / 1e6)

    if avx_avg := values_avx.filter(lambda t: t.infile == file).average():
        plot_basic_avx.append(avx_avg.total_time() / 1e6)

    if avx_avg := values_avx_st.filter(lambda t: t.infile == file).average():
        plot_basic_avx_st.append(avx_avg.total_time() / 1e6)

basic_first_avg = (plot_basic_scalar[0] + plot_basic_avx[0]) / 2
plot_basic_linear: list[float] = list(map(lambda v: v * basic_first_avg, basic_scaling))
plot_basic_pareas = [
    (52.238+53.444)/2,
    (86.748+89.757)/2,
    (125.37+142.89)/2,
    (175.79+192.90)/2,
    (164.20+193.60)/2,
    (410.03+419.91)/2,
    (1342.2+1364.5)/2
]

plot_funclen_scalar: list[float] = []
plot_funclen_avx: list[float] = []
plot_funclen_avx_st: list[float] = []
for file in schema_funclen:
    if scalar_avg := values_scalar.filter(lambda t: t.infile == file).average():
        plot_funclen_scalar.append(scalar_avg.total_time() / 1e6)

    if avx_avg := values_avx.filter(lambda t: t.infile == file).average():
        plot_funclen_avx.append(avx_avg.total_time() / 1e6)

    if avx_avg := values_avx_st.filter(lambda t: t.infile == file).average():
        plot_funclen_avx_st.append(avx_avg.total_time() / 1e6)
funclen_first_avg = plot_funclen_avx[0] / 50
plot_funclen_linear: list[float] = list(map(lambda v: v * funclen_first_avg, funclen_count))
plot_funclen_pareas = [
    (44.030+45.134)/2,
    (37.112+40.007)/2,
    (32.998+34.962)/2,
    (38.112+42.801)/2,
    (41.598+43.011)/2,
    (67.680+68.961)/2,
    (105.87+109.50)/2,
    (216.65+230.40)/2
]

plot_shape_avx: list[float] = []
plot_shape_st: list[float] = []
for file in schema_shape:
    if avx_avg := values_avx.filter(lambda t: t.infile == file).average():
        plot_shape_avx.append(avx_avg.total_time() / 1e6)

    if avx_avg := values_avx_st.filter(lambda t: t.infile == file).average():
        plot_shape_st.append(avx_avg.total_time() / 1e6)
shape_first_avg = (plot_shape_avx[0] + plot_shape_st[0]) / 2
plot_shape_linear: list[float] = list(map(lambda v: v* shape_first_avg, shape_depth))
plot_shape_pareas = [
    78.87,
    957.584,
    45568.068,
    121399.471,
    110620.601,
    97193.301
]

fig, ax = plt.subplots()
ax.plot(basic_size, plot_basic_linear, linestyle="--", label="Linear scaling", color="xkcd:grey")
ax.plot(basic_size, plot_basic_pareas, marker="o", linestyle="--", label="Pareas (RTX3090)", color="xkcd:green")
ax.plot(basic_size[:len(plot_basic_scalar)], plot_basic_scalar, marker="o", label="Scalar", color="xkcd:red")
ax.plot(basic_size, plot_basic_avx_st, marker="o", label="SIMD (1 thread)", color="xkcd:blue")
ax.plot(basic_size, plot_basic_avx, marker="o", label="SIMD (32 threads)", color="xkcd:azure")

ax.set_yscale("log", base=10)
ax.set_xscale("log", base=10)
ax.set_xlabel("Input Size (bytes)")
ax.set_ylabel("Total Execution Time (ms)")

ax.tick_params(axis="both", which="both", direction="in")
ax.tick_params(axis="both", which="both", right=True, top=True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left")
fig.savefig(cast(str, outdir / Path("basic_scalar.pdf")), bbox_inches = "tight", pad_inches = 0.1)

plt.close()

fig, ax = plt.subplots()
ax.plot(funclen_count, plot_funclen_linear, linestyle="--", label="Linear scaling", color="xkcd:grey")
ax.plot(funclen_count, plot_funclen_pareas, linestyle="--", marker="o", label="Pareas (RTX3090)", color="xkcd:green")
ax.plot(funclen_count, plot_funclen_scalar, marker="o", label="Scalar", color="xkcd:red")
ax.plot(funclen_count, plot_funclen_avx_st, marker="o", label="SIMD (1 thread)", color="xkcd:blue")
ax.plot(funclen_count, plot_funclen_avx, marker="o", label="SIMD (32 threads)", color="xkcd:azure")

ax.set_yscale("log", base=10)
ax.set_xscale("log", base=2)
ax.set_xlabel("Function Count")
ax.set_ylabel("Total Execution Time (ms)")

ax.set_xticks(funclen_count, funclen_count)

ax.tick_params(axis="both", which="both", direction="in")
ax.tick_params(axis="both", which="both", right=True, top=True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="center right")
fig.savefig(cast(str, outdir / Path("funclen_scalar.pdf")), bbox_inches = "tight", pad_inches = 0.1)

plt.close()
fig, ax = plt.subplots()
ax.plot(shape_depth, plot_shape_linear, linestyle="--", label="Linear scaling", color="xkcd:grey")
ax.plot(shape_depth, plot_shape_pareas, marker="o", linestyle="--", label="Pareas (RTX3090)", color="xkcd:green")
ax.plot(shape_depth, plot_shape_st, marker="o", label="SIMD (1 thread)", color="xkcd:blue")
ax.plot(shape_depth, plot_shape_avx, marker="o", label="SIMD (32 threads)", color="xkcd:azure")

ax.set_yscale("log", base=10)
ax.set_xlabel("AST Depth")
ax.set_ylabel("Total Execution Time (ms)")

ax.set_xticks(shape_depth, shape_depth)

ax.tick_params(axis="both", which="both", direction="in")
ax.tick_params(axis="both", which="both", right=True, top=True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left")
fig.savefig(cast(str, outdir / Path("shape.pdf")), bbox_inches = "tight", pad_inches = 0.1)
