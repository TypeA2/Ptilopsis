# Contents

This is an excerpt from the codegen branch of the Pareas project. This code
can parse a Pareas input file into an AST and subsequently convert this into
an inverted tree format (class DepthTree).

This code was authored by Marcel Huijben for his MSc thesis.

Also included is a series of test input files in the directory `testfiles`.


## Build

To build this program a C++-20 capable compiler is required, together with
[Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org).
Compiling the project is as easy as:

```
mkdir build
cd build
meson setup
ninja
```

