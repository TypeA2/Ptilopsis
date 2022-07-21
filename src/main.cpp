#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>

#include <codegen/lexer.hpp>
#include <codegen/parser.hpp>
#include <codegen/astnode.hpp>
#include <codegen/exception.hpp>
#include <codegen/depthtree.hpp>
#include <codegen/symtab.hpp>
#include <codegen/treeproperties.hpp>

#include "codegen/rv_generator.hpp"

#include "utils.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
      std::cerr << "usage: " << argv[0] << " <input>\n";
      return EXIT_FAILURE;
    }

    const char *input_path = argv[1];

    try {
        std::ifstream input(input_path);
        if(!input) {
            std::cerr << "Failed to open file " << input_path << std::endl;
            return EXIT_FAILURE;
        }

        Lexer lexer(input);
        SymbolTable symtab;
        Parser parser(lexer, symtab);

        std::unique_ptr<ASTNode> node(parser.parse());
        node->resolveType();

        /* We now have a pointer-linked AST. Show some properties. */
        if (false) {
            TreeProperties props(node.get());
            std::cout << "Number of nodes: " << props.getNodeCount() << std::endl;
            std::cout << "Tree width: " << props.getWidth() << std::endl;
            std::cout << "Tree height: " << props.getDepth() << std::endl;
            std::cout << "Num functions: " << props.getFunctions() << std::endl;
            std::cout << "Max function length: " << props.getMaxFuncLen() << std::endl;
        }

        /* Convert to an inverted tree */
        DepthTree depth_tree(node.get());
        
        rv_generator_st gen{ depth_tree };

        // node->print(std::cout);

        auto begin = std::chrono::steady_clock::now();
        gen.process();
        auto end = std::chrono::steady_clock::now();

        // gen.print(std::cout);

        std::cout << "Processing done in " << (end - begin) << '\n';
    }
    catch(const ParseException& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
