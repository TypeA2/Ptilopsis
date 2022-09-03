# Ptilopsis

SIMD-based Pareas derivative.

## Building

pareas-ast is modified to produce a static library, which is linked to this project.

```
meson setup build
cd build
ninja
```

#### Debug mode
By default debug mode traces the duration of steps of the code generation process. Define `NOTRACE` to disable this.
```
meson setup debug --buildtype debug
cd debug
ninja
```