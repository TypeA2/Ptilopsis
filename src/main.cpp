#include <iostream>
#include <fstream>
#include <memory>

#include <codegen/lexer.hpp>
#include <codegen/parser.hpp>
#include <codegen/astnode.hpp>
#include <codegen/exception.hpp>
#include <codegen/depthtree.hpp>
#include <codegen/symtab.hpp>
#include <codegen/treeproperties.hpp>

#include "codegen/rv_generator.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
      std::cerr << "Please provide a single input file.\n"
                   "usage: " << argv[0] << " <input-file>\n";
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
        TreeProperties props(node.get());
        std::cout << "Number of nodes: " << props.getNodeCount() << std::endl;
        std::cout << "Tree width: " << props.getWidth() << std::endl;
        std::cout << "Tree height: " << props.getDepth() << std::endl;
        std::cout << "Num functions: " << props.getFunctions() << std::endl;
        std::cout << "Max function length: " << props.getMaxFuncLen() << std::endl;

        /* Convert to an inverted tree */
        DepthTree depth_tree(node.get());
        
        rv_generator gen{ depth_tree };

        gen.print(std::cout);
    }
    catch(const ParseException& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
