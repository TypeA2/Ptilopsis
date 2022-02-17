project(
    'pareas',
    ['c', 'cpp'],
    version: '0.0.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
    ]
)

if meson.get_compiler('cpp').get_id() == 'msvc'
	add_project_arguments(
		[ '/std:c++latest', '/DVC_EXTRALEAN', '/DNOMINMAX', ],
		language: ['c', 'cpp'],
	)
else
	add_project_arguments(
	    ['-Wno-unused-parameter', '-Wno-ignored-qualifiers', '-std=c++20'],
	    language: ['c', 'cpp'],
	)
endif

# Final executable
sources = [
    'src/codegen/astnode.cpp',
    'src/codegen/datatype.cpp',
    'src/codegen/depthtree.cpp',
    'src/codegen/lexer.cpp',
    'src/codegen/parser.cpp',
    'src/codegen/symtab.cpp',
    'src/codegen/treeproperties.cpp',
    'src/codegen/token.cpp',
]

includes = [ include_directories('src'), include_directories('include') ]

executable(
    'pareas-ast',
    [ sources, 'src/main.cpp' ],
    install: true,
    build_by_default: true,
    include_directories: includes,
)

# Expose static library version
libpareas_ast = static_library(
    'libpareas-ast',
    sources,
    include_directories: includes
)

# Also expose dependency
libpareas_ast_dep = declare_dependency(
    include_directories: include_directories('include', is_system: true),
    link_with: libpareas_ast
)
