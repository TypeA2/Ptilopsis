#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt

class Test:
    name: str
    values: list[int] = []

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __str__(self):
        return f"<Test name=\"{self.name}\" values=[{len(self.values)} elements]>"

    def __repr__(self):
        return str(self)

tests: list[Test] = []
for line in sys.stdin:
    name, *values = line.rstrip().rstrip(",").split(",")
    
    tests.append(Test(name, list(map(int, values))))

fig = plt.figure()
ax = plt.axes()

for test in tests:
    plt.plot(test.values, label=test.name)

ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Time (ns)")
ax.set_xlim([0, None])
ax.set_ylim([0, None])

if len(sys.argv) > 1:
    fig.savefig(sys.argv[1])
else:
    plt.show()
