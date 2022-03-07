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
```
meson setup debug --buildtype debug
cd debug
ninja
```

### XGetopt
XGetopt is a public domain `getopt` implementation written by Hans Dietrich <[hdietrich2@hotmail.com](mailto:hdietrich2@hotmail.com)>. It is used on the Win32 platform in a slightly modified form.

### Exceptions:
TODO: Check [P2544R0](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2544r0.html)
