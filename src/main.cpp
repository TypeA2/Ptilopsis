#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <filesystem>

#include <codegen/lexer.hpp>
#include <codegen/parser.hpp>
#include <codegen/astnode.hpp>
#include <codegen/exception.hpp>
#include <codegen/depthtree.hpp>
#include <codegen/symtab.hpp>
#include <codegen/treeproperties.hpp>

#include <cxxopts.hpp>

#include "codegen/rv_generator.hpp"
#include "codegen/rv_generator_avx.hpp"

#include "utils.hpp"

int main(int argc, char** argv) {
   // if (argc < 2) {
   //   std::cerr << "usage: " << argv[0] << " <input> [-S] [-o <outfile>]\n";
   //   return EXIT_FAILURE;
   // }
    std::string infile;
    std::string outfile;
    bool output_asm = false;

    cxxopts::Options options("ptilopsis", "Rhine birb");
    options.add_options()
        ("S", "Output assembly instead of machine code", cxxopts::value<bool>()->default_value("false"))
        ("o", "output filename (default: ./c.out)", cxxopts::value<std::string>(), "<outfile>")
        ("infile", "Input filename", cxxopts::value<std::string>())
        ("h,help", "Print usage");

    options.parse_positional("infile");
    options.custom_help("<input> [-S] [-o <outfile>]");
    options.positional_help("");

    try {
        auto res = options.parse(argc, argv);

        if (res.count("help")) {
            std::cout << options.help() << '\n';
            return EXIT_SUCCESS;
        }

        infile = res["infile"].as<std::string>();
        output_asm = res["S"].as<bool>();

        if (res.count("o") == 0) {
            outfile = output_asm ? std::filesystem::path(infile).filename().replace_extension(".s").string() : "c.out";
        } else {
            outfile = res["o"].as<std::string>();
        }
        
    } catch (const cxxopts::OptionException& e) {
        std::cerr << e.what() << '\n';
        std::cerr << options.help() << '\n';
        return EXIT_FAILURE;
    }

    try {
        std::ifstream input(infile);
        if(!input) {
            std::cerr << "Failed to open file " << infile << std::endl;
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

        node->print(std::cout);

        auto begin = std::chrono::steady_clock::now();
        gen.process();
        auto end = std::chrono::steady_clock::now();

        gen.print(std::cout);

        std::cout << "Processing done in " << (end - begin) << '\n';

        std::ofstream output(outfile, std::ios::binary);
        if (output_asm) {
            gen.to_asm(output);
        } else {
            gen.to_binary(output);
        }
    }
    catch(const ParseException& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
