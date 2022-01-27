project(
    'ptilopsis',
    ['c', 'cpp'],
    version: '1.0.0',
    default_options: [
        'warning_level=3',
        'buildtype=release'
    ]
)

is_msvc = meson.get_compiler('cpp').get_id() == 'msvc'

if is_msvc
    add_project_arguments(
        [ '/arch:AVX2', '/std:c++latest', '/DVC_EXTRALEAN', '/DNOMINMAX', ],
        language: ['c', 'cpp']
    )
else
    add_project_arguments(
        [ '-mavx2', '-std=c++20' ],
        language: ['c', 'cpp']
    )
endif

# Use static lib version of pareas-ast as dependency
pareast_proj = subproject('pareas-ast', default_options: ['warning_level=0'])
libpareas_ast_dep = pareast_proj.get_variable('libpareas_ast_dep')
threads_dep = dependency('threads')

ptilopsis_sources = [
    'src/codegen/rv_generator.cpp',
    'src/main.cpp',
]

prefix_sum_sources = [
    'benchmarks/prefix_sum.cpp',
]

if is_msvc
    prefix_sum_sources += 'src/XGetopt.cpp'
endif

includes = [ include_directories('include') ]

executable(
    'ptilopsis',
    ptilopsis_sources,
    dependencies: libpareas_ast_dep,
    install: true,
    build_by_default: true,
    include_directories: includes
)

executable(
    'prefix_sum',
    prefix_sum_sources,
    dependencies: threads_dep,
    include_directories: includes
)

executable(
    'test_shifts',
    [ 'benchmarks/test_shifts.cpp' ],
    include_directories: includes
)
