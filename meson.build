project(
    'ptilopsis',
    ['c', 'cpp'],
    version: '1.0.0',
    default_options: [
        'warning_level=2',
        'buildtype=release',
		'fmt:warning_level=0'
    ]
)

is_msvc = meson.get_compiler('cpp').get_id() == 'msvc'

if is_msvc
    add_project_arguments(
        [
            '/arch:AVX2',
            '/std:c++latest',
            '/DVC_EXTRALEAN', '/DNOMINMAX',
            '/external:anglebrackets', '/external:W3'
        ],
        language: ['c', 'cpp']
    )
else
    add_project_arguments(
        [ '-mavx2', '-std=c++20' ],
        language: ['c', 'cpp']
    )
endif

threads_dep = dependency('threads')

# Use static lib version of pareas-ast as dependency
libpareas_ast_dep = subproject('pareas-ast', default_options: ['warning_level=0']).get_variable('libpareas_ast_dep')
fmt_header_only_dep = subproject('fmt', default_options: ['warning_level=0']).get_variable('fmt_header_only_dep')
magic_enum_dep = subproject('magic_enum', default_options: ['warning_level=0']).get_variable('magic_enum_dep')

ptilopsis_sources = [
    'src/disassembler.cpp',
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
    dependencies: [ threads_dep, libpareas_ast_dep, fmt_header_only_dep, magic_enum_dep ],
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
