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
    int threads = -1;
    int cv = false;
    bool altblock = false;
    uint32_t mininstr = 0;

    cxxopts::Options options("ptilopsis", "Rhine birb");
    options.add_options()
        ("S", "Output assembly instead of machine code", cxxopts::value<bool>()->default_value("false"))
        ("o", "output filename (default: ./c.out, - for stdout)", cxxopts::value<std::string>(), "<outfile>")
        ("infile", "Input filename, - for s tdin", cxxopts::value<std::string>())
        ("h,help", "Print usage")
        ("d,debug", "Print debugging info", cxxopts::value<bool>()->default_value("false"))
        ("s,simple", "Use simple compiler instead of AVX compiler", cxxopts::value<bool>()->default_value("false"))
        ("f,tree", "Print tree info", cxxopts::value<bool>()->default_value("false"))
        ("t,threads", "Number of threads to use in the AVX implementation (-1 means total - 1)", cxxopts::value<int>()->default_value("-1"))
        ("m,sync", "Synchronization mechanism (0 = barrier, 1 = custom spinlock)", cxxopts::value<int>()->default_value("0"))
        ("b,altblock", "Use alternate loop blocking method", cxxopts::value<bool>()->default_value("false"))
        ("i,mininstr", "Minimum number of 8-instruction AVX vectors (or 4x the minimum number of nodes) to process per thread", cxxopts::value<uint32_t>()->default_value("0"));

    options.parse_positional("infile");
    options.custom_help("<infile> [-S] [-o <outfile>] [-d] [-s] [-f] [-t <threadnum>] [-m <syncmode>] [-b] -[i mininstr]");
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
        threads = res["threads"].as<int>();
        cv = res["sync"].as<int>();
        altblock = res["altblock"].as<bool>();
        mininstr = res["mininstr"].as<uint32_t>();
        
    } catch (const cxxopts::OptionException& e) {
        std::cerr << e.what() << '\n';
        std::cerr << options.help() << '\n';
        return EXIT_FAILURE;
    }

#ifdef _MSC_VER
    if (!SetProcessAffinityMask(GetCurrentProcess(), (1ui32 << threads) - 1)) {
        std::cerr << "Failed to set processor affinity!\n";
        return EXIT_FAILURE;
    }
#endif

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
            gen = std::make_unique<rv_generator_avx>(depth_tree, threads, cv, altblock, mininstr);
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
        
    } catch(const ParseException& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
