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

parser.add_argument("infile", type=argparse.FileType("r", encoding="utf-8"), help="Input file .csv")
parser.add_argument("outdir", type=outdir_check, help="Output image directory")

@dataclass
class TestResult:
    threads: int
    alt_block: bool
    lock_type: int
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
        threads, alt_block, lock_type, infile, values = line.split(",", 4)
        
        durations = list(map(int, values.lstrip("[").rstrip("]").split(", ")[:-1]))

        return TestResult(
            int(threads), bool(int(alt_block)), int(lock_type), infile,
            durations[0], durations[1], durations[2], durations[3], durations[4], durations[5], durations[6]
        )

    def __str__(self) -> str:
        pre: str = f"{self.threads},{1 if self.alt_block else 0},{self.lock_type},{self.infile}"
        post: str = f"{self.preprocess},{self.isn_cnt},{self.isn_gen},{self.optimize},{self.regalloc},{self.fix_jumps},{self.postprocess}"
        return f"{pre},{post}"

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

    @property
    def thread_counts(self) -> list[int]:
        res: list[int] = []
        lookup: set[int] = set()
        v: TestResult
        for v in self:
            if (t := v.threads) not in lookup:
                lookup.add(t)
                res.append(t)

        return res

    def average(self) -> TestResult:
        res = TestResult(
            self[0].threads, self[0].alt_block, self[0].lock_type, self[0].infile,
            0, 0, 0, 0, 0, 0, 0
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

infile: TextIO = args.infile
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
    "shape/6.in"
]

for file in schema:
    try:
        Path(outdir / file).parent.mkdir(parents=True)
    except FileExistsError:
        pass

values = TestResultArray()
for line in infile:
    values.append(line.rstrip("\n"))

names = {
    "preprocess": "Preprocessing",
    "isn_cnt": "Instruction Counting",
    "isn_gen": "Instruction Generation",
    "optimize": "Optimization",
    "regalloc": "Register Allocation",
    "fix_jumps": "Jump Fixing",
    "postprocess": "Postprocessing"
}

colors = [
    f"xkcd:{col}" for col in [ "aqua", "coral", "beige", "grey", "purple", "darkgreen", "blue" ]
]

threads = values.thread_counts
bar_indices = numpy.arange(len(threads))
width = 0.15
dist = 0.05
indices = [
    bar_indices - width - 3 * dist,
    bar_indices - width/2 - dist/2,
    bar_indices + width/2 + dist/2,
    bar_indices + width + 3 * dist
]

indices_arr = numpy.concatenate(indices)

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Computer Modern"],
    "text.usetex": True
})

def select_stage(arr: TestResultArray, alt_block: bool, lock_type: int) -> list[list[float]]:
    arr = arr.filter(lambda t: t.alt_block == alt_block and t.lock_type == lock_type)

    vals = TestResultArray()
    for t in arr.thread_counts:
        vals.append(arr.filter(lambda r: r.threads == t).average())

    return [
        vals.select(lambda t: t.preprocess / 1e6),
        vals.select(lambda t: t.isn_cnt / 1e6),
        vals.select(lambda t: t.isn_gen / 1e6),
        vals.select(lambda t: t.optimize / 1e6),
        vals.select(lambda t: t.regalloc / 1e6),
        vals.select(lambda t: t.fix_jumps / 1e6),
        vals.select(lambda t: t.postprocess / 1e6)
    ]

def plot_stages(ax: matplotlib.axes.Axes, stages: list[list[list[float]]]):
    bottom = [[0. for i in range(len(threads))] for j in range(len(stages))]
    name_keys = list(names.keys())
    for stage in range(len(names)):
        stage = len(names) - stage - 1
        
        vals = [
            stages[i][stage] for i in range(len(stages))
        ]
        ax.bar(indices_arr, numpy.concatenate(vals), width, bottom=numpy.concatenate(bottom), label=name_keys[stage], color=colors[stage])

        for thread_count in range(len(threads)):
            for i in range(len(stages)):
                bottom[i][thread_count] += stages[i][stage][thread_count]


got_legend = False

def plot(selected: TestResultArray, file: str, **kwargs) -> None:
    stages = [
        select_stage(selected, False, 0),
        select_stage(selected, True, 0),
        select_stage(selected, False, 1),
        select_stage(selected, True, 1)
    ]

    fig, ax = plt.subplots()

    ax.set_title(file)
    plot_stages(ax, stages)
    
    ax.get_xaxis().tick_top()
    ax.get_xaxis().set_label_position("top")

    ax.set_xlabel("Threads")
    ax.set_xticks(bar_indices, threads)
    ax.grid(True, "major", "x", zorder=0)
    ax.set_ylabel("Total Execution Time (ms)")

    ax2 = ax.secondary_xaxis("bottom")
    ax2.set_xlabel("Configuration")
    ax2.set_xticks(indices_arr, ["a"] * len(threads) + ["b"] * len(threads) + ["c"] * len(threads) + ["d"] * len(threads))

    name = str(Path(file).stem)

    if "limit_y" in kwargs and kwargs["limit_y"]:
        max_val = 0
        for thread_count in range(len(threads)):
            s0 = 0
            s1 = 0
            for stage in range(7):
                s0 += stages[0][stage][thread_count]
                s1 += stages[1][stage][thread_count]

            if s0 > max_val:
                max_val = s0

            if s1 > max_val:
                max_val = s1
            
        ax.set_ylim(bottom=0, top=max_val * 1.15)
        name = name + "_limit_y"

    global got_legend
    if not got_legend:
        got_legend = True
        legend_fig = pylab.figure(figsize = cast(tuple[float], fig.get_size_inches()))
        handles, labels = ax.get_legend_handles_labels()
        pylab.figlegend(handles[::-1], [names[n] for n in labels[::-1]], loc = "center", fontsize = "xx-large")
        legend_fig.savefig(cast(str, outdir / "legend.pdf"))

    fig.savefig(cast(str, outdir / Path(file).with_stem(name).with_suffix(".pdf")), bbox_inches = "tight", pad_inches = 0)
    plt.close()

def graph(selected: TestResultArray, file: str) -> None:
    with open(outdir / Path(file).with_suffix(".tex"), "w", encoding="utf-8") as outfile:
        outfile.write(r"""\begin{longtable}{|r|r|r|r|r|r|}
    \hline
    \textbf{Threads} & \specialcell{\textbf{Alternate} \\ \textbf{Blocking}}
    &  \textbf{Lock type} &  \specialcell{\textbf{Minimum} \\ \textbf{Runtime (ms)}}
    &  \specialcell{\textbf{Maximum} \\ \textbf{Runtime (ms)}} & \specialcell{\textbf{Standard} \\ \textbf{Deviation (ms)}} \\
    \hline
""")
        for thread_count in threads:
            for lock_type in [0, 1]:
                for alt_block in [True, False]:
                    match = selected.filter(lambda t: t.threads == thread_count and t.lock_type == lock_type and t.alt_block == alt_block)
                    
                    shortest = min(match, key = lambda t: t.total_time())
                    longest = max(match, key = lambda t: t.total_time())

                    outfile.write(f"    {thread_count} & {'Yes' if alt_block else 'No'} & {lock_type} & {shortest.total_time() / 1e6:.3f}")
                    outfile.write(f" & {longest.total_time() / 1e6:.3f} & {numpy.std(match.select(lambda t: t.total_time())) / 1e6:.3f} \\\\\n    \\hline\n")

        outfile.write("    \\caption{\\TableCaption}\n")
        outfile.write("    \\label{\\TableLabel}\n")
        outfile.write("\\end{longtable}\n")

for file in schema:
    selected = values.filter(lambda t: t.infile == file)

    plot(selected, file)
    plot(selected, file, limit_y=True)

    graph(selected, file)

    print(file)

