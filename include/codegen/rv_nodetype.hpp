#pragma once

#include <array>
#include <iostream>

#include <magic_enum.hpp>

#include <codegen/astnode.hpp>

using namespace magic_enum::bitwise_operators;
using namespace magic_enum::ostream_operators;

/* Separate node type definitions because
 * at this point they're not typechecked anymore
 */
enum class rv_node_type : uint8_t {
    invalid,
    statement_list,
    empty,
    expression,

    add_expr,
    sub_expr,
    mul_expr,
    div_expr,
    mod_expr,
    bitand_expr,
    bitor_expr,
    bitxor_expr,
    lshift_expr,
    rshift_expr,
    urshift_expr,
    logic_and_expr,
    logic_or_expr,

    bitnot_expr,
    logic_not_expr,
    neg_expr,

    literal_expr,
    cast_expr,
    deref_expr,
    assign_expr,
    decl_expr,
    id_expr,
    while_dummy,
    func_decl_dummy,
    return_statement,

    func_decl,
    func_arg_list,

    func_call_expression,
    func_call_arg_list,

    /*
     * Branch operations have their upper 4 bits set to 0b1101
     * The lowest bit represents the child index of the condition node:
     *  - if and if_else have their condition as the first node
     *  - while has it's condition as a second node
     */
    if_statement      = 0b11010000,
    while_statement   = 0b11010001,
    if_else_statement = 0b11010010,
    

    /*
     * Comparison operators have the upper 4 bits set to 0b1110
     */
    eq_expr  = 0b11100000,
    neq_expr = 0b11100001,
    lt_expr  = 0b11100010,
    gt_expr  = 0b11100011,
    lte_expr = 0b11100100,
    gte_expr = 0b11100101,

    /*
     * Func args will always have the upper 4 bits set, and the 3rd bit is set if it's a call.
     * Lower 2 bits determine the type of argument (NOT the type _of_ the argument).
     *
     * This way a bitwise AND with func_arg identifies any arguments
     */
    func_arg                   = 0b11110000,
    func_arg_float_as_int      = 0b11110001,
    func_arg_on_stack          = 0b11110010,
    func_call_arg              = 0b11110100,
    func_call_arg_float_as_int = 0b11110101,
    func_call_arg_on_stack     = 0b11110110,
    
};

using rv_node_type_t = std::underlying_type_t<rv_node_type>;
using data_type_t = std::underlying_type_t<DataType>;

constexpr size_t max_node_types = 1 << (sizeof(rv_node_type_t) * std::numeric_limits<rv_node_type_t>::digits);

template <>
struct magic_enum::customize::enum_range<rv_node_type> {
    static constexpr int min = 0;
    static constexpr int max = max_node_types;
};

template <typename T> requires std::is_enum_v<T>
constexpr auto as_index(T v) {
    // return magic_enum::enum_integer<T>
    return static_cast<uint32_t>(v);
}

constexpr rv_node_type operator+(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) + rhs);
}

constexpr rv_node_type& operator+=(rv_node_type& lhs, rv_node_type_t rhs) {
    return (lhs = lhs + rhs);
}

/*
constexpr rv_node_type operator&(rv_node_type lhs, rv_node_type rhs) {
    return static_cast<rv_node_type>(
        static_cast<rv_node_type_t>(lhs) & static_cast<rv_node_type_t>(rhs));
}
*/

constexpr rv_node_type operator&(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) & rhs);
}

constexpr rv_node_type& operator&=(rv_node_type& lhs, rv_node_type_t rhs) {
    return (lhs = static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) & rhs));
}

/*
constexpr rv_node_type operator|(rv_node_type lhs, rv_node_type rhs) {
    return static_cast<rv_node_type>(
        static_cast<rv_node_type_t>(lhs) | static_cast<rv_node_type_t>(rhs));
}
*/

constexpr rv_node_type operator|(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) | rhs);
}

constexpr rv_node_type& operator|=(rv_node_type& lhs, rv_node_type_t rhs) {
    return (lhs = static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) | rhs));
}

constexpr bool operator==(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type_t>(lhs) == rhs;
}

constexpr data_type_t operator&(DataType lhs, data_type_t rhs) {
    return static_cast<data_type_t>(lhs) & rhs;
}

/* Similar to NODE_COUNT_TABLE in Pareas */
constexpr auto generate_size_mapping() {
    using enum rv_node_type;

    /* res[x][y] = node size for node type x with return type y*/
    std::array<std::array<uint32_t, magic_enum::enum_count<DataType>()>, max_node_types> res{};

    /* Hardcode all values like this because we dont have designated array initializers */
    res[as_index(invalid)]        = { 0, 0, 0, 0, 0, 0 };
    res[as_index(statement_list)] = { 0, 0, 0, 0, 0, 0 };
    res[as_index(empty)]          = { 0, 0, 0, 0, 0, 0 };
    res[as_index(expression)]     = { 0, 0, 0, 0, 0, 0 };

    res[as_index(add_expr)]       = { 1, 1, 1, 1, 1, 1 };
    res[as_index(sub_expr)]       = { 1, 1, 1, 1, 1, 1 };
    res[as_index(mul_expr)]       = { 1, 1, 1, 1, 1, 1 };
    res[as_index(div_expr)]       = { 1, 1, 1, 1, 1, 1 };
    res[as_index(mod_expr)]       = { 1, 1, 1, 1, 1, 1 };
    res[as_index(bitand_expr)]    = { 1, 1, 1, 1, 1, 1 };
    res[as_index(bitor_expr)]     = { 1, 1, 1, 1, 1, 1 };
    res[as_index(bitxor_expr)]    = { 1, 1, 1, 1, 1, 1 };
    res[as_index(lshift_expr)]    = { 1, 1, 1, 1, 1, 1 };
    res[as_index(rshift_expr)]    = { 1, 1, 1, 1, 1, 1 };
    res[as_index(urshift_expr)]   = { 1, 1, 1, 1, 1, 1 };
    res[as_index(logic_and_expr)] = { 0, 0, 0, 0, 0, 0 };
    res[as_index(logic_or_expr)]  = { 0, 0, 0, 0, 0, 0 };

    res[as_index(bitnot_expr)]    = { 1, 1, 1, 1, 1, 1 };
    res[as_index(logic_not_expr)] = { 1, 1, 1, 1, 1, 1 };
    res[as_index(neg_expr)]       = { 1, 1, 1, 1, 1, 1 };

    res[as_index(literal_expr)]     = { 0, 0, 2, 3, 0, 0 };
    res[as_index(cast_expr)]        = { 1, 1, 1, 1, 1, 1 };
    res[as_index(deref_expr)]       = { 1, 1, 1, 1, 1, 1 };
    res[as_index(assign_expr)]      = { 2, 2, 2, 2, 2, 2 };
    res[as_index(decl_expr)]        = { 1, 1, 1, 1, 1, 1 };
    res[as_index(id_expr)]          = { 1, 1, 1, 1, 1, 1 };
    res[as_index(while_dummy)]      = { 0, 0, 0, 0, 0, 0 };
    res[as_index(func_decl_dummy)]  = { 6, 6, 6, 6, 6, 6 };
    res[as_index(return_statement)] = { 2, 2, 2, 2, 2, 2 };

    res[as_index(func_decl)]     = { 6, 6, 6, 6, 6, 6 };
    res[as_index(func_arg_list)] = { 0, 0, 0, 0, 0, 0 };

    res[as_index(func_call_expression)] = { 3, 3, 3, 3, 3, 3 };
    res[as_index(func_call_arg_list)]   = { 0, 0, 0, 0, 0, 0 };

    res[as_index(if_statement)]      = { 0, 0, 0, 0, 0, 0 };
    res[as_index(while_statement)]   = { 1, 1, 1, 1, 1, 1 };
    res[as_index(if_else_statement)] = { 0, 0, 0, 0, 0, 0 };

    res[as_index(eq_expr)]  = { 0, 0, 2, 1, 0, 0 };
    res[as_index(neq_expr)] = { 2, 2, 2, 2, 2, 2 };
    res[as_index(lt_expr)]  = { 1, 1, 1, 1, 1, 1 };
    res[as_index(gt_expr)]  = { 1, 1, 1, 1, 1, 1 };
    res[as_index(lte_expr)] = { 0, 0, 2, 1, 0, 0 };
    res[as_index(gte_expr)] = { 0, 0, 2, 1, 0, 0 };

    res[as_index(func_arg)]                   = { 1, 1, 1, 1, 1, 1 };
    res[as_index(func_arg_float_as_int)]      = { 1, 1, 1, 1, 1, 1 };
    res[as_index(func_arg_on_stack)]          = { 2, 2, 2, 2, 2, 2 };
    res[as_index(func_call_arg)]              = { 0, 0, 0, 0, 0, 0 };
    res[as_index(func_call_arg_float_as_int)] = { 0, 0, 0, 0, 0, 0 };
    res[as_index(func_call_arg_on_stack)]     = { 0, 0, 0, 0, 0, 0 };

    return res;
}

constexpr auto node_size_mapping = generate_size_mapping();

constexpr rv_node_type pareas_to_rv_nodetype[] {
    /* [NodeType::INVALID]            = */ rv_node_type::invalid,
    /* [NodeType::STATEMENT_LIST]     = */ rv_node_type::statement_list,
    /* [NodeType::EMPTY_STAT]         = */ rv_node_type::empty,
    /* [NodeType::FUNC_DECL]          = */ rv_node_type::func_decl,
    /* [NodeType::FUNC_ARG]           = */ rv_node_type::func_arg,
    /* [NodeType::FUNC_ARG_LIST]      = */ rv_node_type::func_arg_list,
    /* [NodeType::EXPR_STAT]          = */ rv_node_type::expression,
    /* [NodeType::IF_STAT]            = */ rv_node_type::if_statement,
    /* [NodeType::IF_ELSE_STAT]       = */ rv_node_type::if_else_statement,
    /* [NodeType::WHILE_STAT]         = */ rv_node_type::while_statement,
    /* [NodeType::FUNC_CALL_EXPR]     = */ rv_node_type::func_call_expression,
    /* [NodeType::FUNC_CALL_ARG]      = */ rv_node_type::func_call_arg,
    /* [NodeType::FUNC_CALL_ARG_LIST] = */ rv_node_type::func_call_arg_list,
    /* [NodeType::ADD_EXPR]           = */ rv_node_type::add_expr,
    /* [NodeType::SUB_EXPR]           = */ rv_node_type::sub_expr,
    /* [NodeType::MUL_EXPR]           = */ rv_node_type::mul_expr,
    /* [NodeType::DIV_EXPR]           = */ rv_node_type::div_expr,
    /* [NodeType::MOD_EXPR]           = */ rv_node_type::mod_expr,
    /* [NodeType::BITAND_EXPR]        = */ rv_node_type::bitand_expr,
    /* [NodeType::BITOR_EXPR]         = */ rv_node_type::bitor_expr,
    /* [NodeType::BITXOR_EXPR]        = */ rv_node_type::bitxor_expr,
    /* [NodeType::LSHIFT_EXPR]        = */ rv_node_type::lshift_expr,
    /* [NodeType::RSHIFT_EXPR]        = */ rv_node_type::rshift_expr,
    /* [NodeType::URSHIFT_EXPR]       = */ rv_node_type::urshift_expr,
    /* [NodeType::LAND_EXPR]          = */ rv_node_type::logic_and_expr,
    /* [NodeType::LOR_EXPR]           = */ rv_node_type::logic_or_expr,
    /* [NodeType::EQ_EXPR]            = */ rv_node_type::eq_expr,
    /* [NodeType::NEQ_EXPR]           = */ rv_node_type::neq_expr,
    /* [NodeType::LESS_EXPR]          = */ rv_node_type::lt_expr,
    /* [NodeType::GREAT_EXPR]         = */ rv_node_type::gt_expr,
    /* [NodeType::LESSEQ_EXPR]        = */ rv_node_type::lte_expr,
    /* [NodeType::GREATEQ_EXPR]       = */ rv_node_type::gte_expr,
    /* [NodeType::BITNOT_EXPR]        = */ rv_node_type::bitnot_expr,
    /* [NodeType::LNOT_EXPR]          = */ rv_node_type::logic_not_expr,
    /* [NodeType::NEG_EXPR]           = */ rv_node_type::neg_expr,
    /* [NodeType::LIT_EXPR]           = */ rv_node_type::literal_expr,
    /* [NodeType::CAST_EXPR]          = */ rv_node_type::cast_expr,
    /* [NodeType::DEREF_EXPR]         = */ rv_node_type::deref_expr,
    /* [NodeType::ASSIGN_EXPR]        = */ rv_node_type::assign_expr,
    /* [NodeType::DECL_EXPR]          = */ rv_node_type::decl_expr,
    /* [NodeType::ID_EXPR]            = */ rv_node_type::id_expr,
    /* [NodeType::WHILE_DUMMY]        = */ rv_node_type::while_dummy,
    /* [NodeType::FUNC_DECL_DUMMY]    = */ rv_node_type::func_decl_dummy,
    /* [NodeType::RETURN_STAT]        = */ rv_node_type::return_statement,
};


