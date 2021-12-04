project(
    'ptilopsis',
    ['c', 'cpp'],
    version: '1.0.0',
    default_options: [
        'cpp_std=c++20',
        'warning_level=3',
        'buildtype=release',
    ]
)

# Use static lib version of pareas-ast as dependency
pareast_proj = subproject('pareas-ast', default_options: ['warning_level=0'])
libpareas_ast_dep = pareast_proj.get_variable('libpareas_ast_dep')

sources = [
    'src/main.cpp'
]

includes = [ include_directories('include') ]

executable(
    'ptilopsis',
    sources,
    dependencies: libpareas_ast_dep,
    install: true,
    build_by_default: true,
    include_directories: includes
)
