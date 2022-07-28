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
    std::string infile;
    std::string outfile;
    bool output_asm = false;
    bool debug = false;
    bool simple = false;
    bool tree = false;

    cxxopts::Options options("ptilopsis", "Rhine birb");
    options.add_options()
        ("S", "Output assembly instead of machine code", cxxopts::value<bool>()->default_value("false"))
        ("o", "output filename (default: ./c.out, - for stdout)", cxxopts::value<std::string>(), "<outfile>")
        ("infile", "Input filename, - for s tdin", cxxopts::value<std::string>())
        ("h,help", "Print usage")
        ("d,debug", "Print debugging info", cxxopts::value<bool>()->default_value("false"))
        ("s,simple", "Use simple compiler instead of AVX compiler", cxxopts::value<bool>()->default_value("false"))
        ("t,tree", "Print tree info", cxxopts::value<bool>()->default_value("false"));

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

        debug = res["debug"].as<bool>();
        simple = res["simple"].as<bool>();
        tree = res["tree"].as<bool>();
        
    } catch (const cxxopts::OptionException& e) {
        std::cerr << e.what() << '\n';
        std::cerr << options.help() << '\n';
        return EXIT_FAILURE;
    }

    try {
        std::unique_ptr<std::istream> input;
        std::unique_ptr<Lexer> lexer;

        if (infile == "-") {
            lexer = std::make_unique<Lexer>(std::cin);
        } else {
            input = std::make_unique<std::ifstream>(infile);

            if (!input->good()) {
                std::cerr << "Failed to open file " << infile << '\n';
                return EXIT_FAILURE;
            }
            lexer = std::make_unique<Lexer>(*input);
        }
        
        SymbolTable symtab;
        Parser parser(*lexer, symtab);

        std::unique_ptr<ASTNode> node(parser.parse());
        node->resolveType();

        /* We now have a pointer-linked AST. Show some properties. */
        if (debug) {
            TreeProperties props(node.get());
            std::cout << "Number of nodes: " << props.getNodeCount() << std::endl;
            std::cout << "Tree width: " << props.getWidth() << std::endl;
            std::cout << "Tree height: " << props.getDepth() << std::endl;
            std::cout << "Num functions: " << props.getFunctions() << std::endl;
            std::cout << "Max function length: " << props.getMaxFuncLen() << std::endl;
        }

        /* Convert to an inverted tree */
        DepthTree depth_tree(node.get());
        
        std::unique_ptr<rv_generator> gen;

        if (simple) {
            gen = std::make_unique<rv_generator_st>(depth_tree);
        } else {
            gen = std::make_unique<rv_generator_avx>(depth_tree);
        }

        if (debug || tree) {
            node->print(std::cout);
        }

        gen->process();

        if (debug || tree) {
            gen->print(std::cout, debug);
        }

        if (outfile == "-") {
            if (output_asm) {
                gen->to_asm(std::cout);
            } else {
                gen->to_binary(std::cout);
            }
        } else {
            std::ofstream output(outfile, std::ios::binary);
            if (output.bad()) {
                std::cout << "Failed to open output file: " << outfile << '\n';
                return EXIT_FAILURE;
            }

            if (output_asm) {
                gen->to_asm(output);
            } else {
                gen->to_binary(output);
            }
        }
        
    }
    catch(const ParseException& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
