#pragma once

#include <codegen/astnode.hpp>

/* Separate node type definitions because
 * at this point they're not typechecked anymore
 */
enum class rv_node_type : uint8_t {
    invalid,
    statement_list,
    empty,
    expression,
    if_statement,
    if_else_statement,
    while_statement,

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
    eq_expr,
    neq_expr,
    lt_expr,
    gt_expr,
    lte_expr,
    gte_expr,
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
     * Func args will always have the 8th bit set, and the 7th bit is set if it's a call.
     * Lower 2 bits determine the type of argument (NOT the type _of_ the argument).
     *
     * This way a bitwise AND with func_arg identifies any arguments
     */
    func_arg                   = 0b10000000,
    func_arg_float_as_int      = 0b10000001,
    func_arg_on_stack          = 0b10000010,
    func_call_arg              = 0b11000000,
    func_call_arg_float_as_int = 0b11000001,
    func_call_arg_on_stack     = 0b11000010,
    
};

using rv_node_type_t = std::underlying_type_t<rv_node_type>;
using data_type_t = std::underlying_type_t<DataType>;

constexpr rv_node_type operator+(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) + rhs);
}

constexpr rv_node_type& operator+=(rv_node_type& lhs, rv_node_type_t rhs) {
    return (lhs = lhs + rhs);
}

constexpr rv_node_type_t operator&(rv_node_type lhs, rv_node_type rhs) {
    return static_cast<rv_node_type_t>(lhs) & static_cast<rv_node_type_t>(rhs);
}

constexpr rv_node_type_t operator&(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type_t>(lhs) & rhs;
}

constexpr rv_node_type& operator&=(rv_node_type& lhs, rv_node_type_t rhs) {
    return (lhs = static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) & rhs));
}

constexpr rv_node_type_t operator|(rv_node_type lhs, rv_node_type rhs) {
    return static_cast<rv_node_type_t>(lhs) | static_cast<rv_node_type_t>(rhs);
}

constexpr rv_node_type_t operator|(rv_node_type lhs, rv_node_type_t rhs) {
    return static_cast<rv_node_type_t>(lhs) | rhs;
}

constexpr rv_node_type& operator|=(rv_node_type& lhs, rv_node_type_t rhs) {
    return (lhs = static_cast<rv_node_type>(static_cast<rv_node_type_t>(lhs) | rhs));
}

constexpr data_type_t operator&(DataType lhs, data_type_t rhs) {
    return static_cast<data_type_t>(lhs) & rhs;
}

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
