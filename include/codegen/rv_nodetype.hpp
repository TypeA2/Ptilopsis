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

constexpr size_t parent_idx_per_node = 3;

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

template <typename T, size_t N>
class mapping_helper {
    std::array<T, N> res{};

    public:
    constexpr mapping_helper() = default;

    [[nodiscard]] constexpr std::array<T, N> get() {
        return std::move(res);
    }

    [[nodiscard]] constexpr T& operator[](uint32_t i) {
        return res[i];
    }

    [[nodiscard]] constexpr T& operator[](rv_node_type t) {
        return res[as_index(t)];
    }
};

template <typename T>
using ArrayForTypes = std::array<T, magic_enum::enum_count<DataType>()>;

/* Similar to NODE_COUNT_TABLE in Pareas */
constexpr auto generate_size_mapping() {
    using enum rv_node_type;

    /* res[x][y] = node size for node type x with return type y*/
    mapping_helper<ArrayForTypes<uint32_t>, max_node_types> res;

    /* Hardcode all values like this because we dont have designated array initializers */
    res[invalid]        = { 0, 0, 0, 0, 0, 0 };
    res[statement_list] = { 0, 0, 0, 0, 0, 0 };
    res[empty]          = { 0, 0, 0, 0, 0, 0 };
    res[expression]     = { 0, 0, 0, 0, 0, 0 };

    res[add_expr]       = { 1, 1, 1, 1, 1, 1 };
    res[sub_expr]       = { 1, 1, 1, 1, 1, 1 };
    res[mul_expr]       = { 1, 1, 1, 1, 1, 1 };
    res[div_expr]       = { 1, 1, 1, 1, 1, 1 };
    res[mod_expr]       = { 1, 1, 1, 1, 1, 1 };
    res[bitand_expr]    = { 1, 1, 1, 1, 1, 1 };
    res[bitor_expr]     = { 1, 1, 1, 1, 1, 1 };
    res[bitxor_expr]    = { 1, 1, 1, 1, 1, 1 };
    res[lshift_expr]    = { 1, 1, 1, 1, 1, 1 };
    res[rshift_expr]    = { 1, 1, 1, 1, 1, 1 };
    res[urshift_expr]   = { 1, 1, 1, 1, 1, 1 };
    res[logic_and_expr] = { 0, 0, 0, 0, 0, 0 };
    res[logic_or_expr]  = { 0, 0, 0, 0, 0, 0 };

    res[bitnot_expr]    = { 1, 1, 1, 1, 1, 1 };
    res[logic_not_expr] = { 1, 1, 1, 1, 1, 1 };
    res[neg_expr]       = { 1, 1, 1, 1, 1, 1 };

    res[literal_expr]     = { 0, 0, 2, 3, 0, 0 };
    res[cast_expr]        = { 1, 1, 1, 1, 1, 1 };
    res[deref_expr]       = { 1, 1, 1, 1, 1, 1 };
    res[assign_expr]      = { 2, 2, 2, 2, 2, 2 };
    res[decl_expr]        = { 1, 1, 1, 1, 1, 1 };
    res[id_expr]          = { 1, 1, 1, 1, 1, 1 };
    res[while_dummy]      = { 0, 0, 0, 0, 0, 0 };
    res[func_decl_dummy]  = { 6, 6, 6, 6, 6, 6 };
    res[return_statement] = { 2, 2, 2, 2, 2, 2 };

    res[func_decl]     = { 6, 6, 6, 6, 6, 6 };
    res[func_arg_list] = { 0, 0, 0, 0, 0, 0 };

    res[func_call_expression] = { 3, 3, 3, 3, 3, 3 };
    res[func_call_arg_list]   = { 0, 0, 0, 0, 0, 0 };

    res[if_statement]      = { 0, 0, 0, 0, 0, 0 };
    res[while_statement]   = { 1, 1, 1, 1, 1, 1 };
    res[if_else_statement] = { 0, 0, 0, 0, 0, 0 };

    res[eq_expr]  = { 0, 0, 2, 1, 0, 0 };
    res[neq_expr] = { 2, 2, 2, 2, 2, 2 };
    res[lt_expr]  = { 1, 1, 1, 1, 1, 1 };
    res[gt_expr]  = { 1, 1, 1, 1, 1, 1 };
    res[lte_expr] = { 0, 0, 2, 1, 0, 0 };
    res[gte_expr] = { 0, 0, 2, 1, 0, 0 };

    res[func_arg]                   = { 1, 1, 1, 1, 1, 1 };
    res[func_arg_float_as_int]      = { 1, 1, 1, 1, 1, 1 };
    res[func_arg_on_stack]          = { 2, 2, 2, 2, 2, 2 };
    res[func_call_arg]              = { 0, 0, 0, 0, 0, 0 };
    res[func_call_arg_float_as_int] = { 0, 0, 0, 0, 0, 0 };
    res[func_call_arg_on_stack]     = { 0, 0, 0, 0, 0, 0 };

    return res.get();
}

constexpr auto node_size_mapping = generate_size_mapping();

/* Our HAS_INSTR_TABLE */
constexpr auto generate_has_instr() {
    using enum rv_node_type;

    mapping_helper<std::array<ArrayForTypes<bool>, 4>, max_node_types> res;

    ArrayForTypes<bool> yes { true,  true,  true,  true,  true,  true  };
    ArrayForTypes<bool> no  { false, false, false, false, false, false };

    res[invalid]        = { no,  no, no, no };
    res[statement_list] = { no,  no, no, no };
    res[empty]          = { no,  no, no, no };
    res[expression]     = { no,  no, no, no };

    res[add_expr]       = { yes, no, no, no };
    res[sub_expr]       = { yes, no, no, no };
    res[mul_expr]       = { yes, no, no, no };
    res[div_expr]       = { yes, no, no, no };
    res[mod_expr]       = { yes, no, no, no };
    res[bitand_expr]    = { yes, no, no, no };
    res[bitor_expr]     = { yes, no, no, no };
    res[bitxor_expr]    = { yes, no, no, no };
    res[lshift_expr]    = { yes, no, no, no };
    res[rshift_expr]    = { yes, no, no, no };
    res[urshift_expr]   = { yes, no, no, no };
    res[logic_and_expr] = { yes, no, no, no };
    res[logic_or_expr]  = { yes, no, no, no };

    res[bitnot_expr]    = { yes, no, no, no };
    res[logic_not_expr] = { yes, no, no, no };
    res[neg_expr]       = { yes, no, no, no };

    res[literal_expr]     = { yes, yes, { false, false, false, true, false, false},  no };
    res[cast_expr]        = { yes, no,  no,  no  };
    res[deref_expr]       = { yes, no,  no,  no  };
    res[assign_expr]      = { yes, yes, no,  no  };
    res[decl_expr]        = { yes, no,  no,  no  };
    res[id_expr]          = { yes, no,  no,  no  };
    res[while_dummy]      = { yes, no,  no,  no  };
    res[func_decl_dummy]  = { yes, yes, yes, yes };
    res[return_statement] = { yes, yes, no,  no  };

    res[func_decl]     = { yes, yes, yes, yes };
    res[func_arg_list] = { no,  no,  no,  no  };

    res[func_call_expression] = { yes, no, { false, false, true, true, false, false}, no };
    res[func_call_arg_list]   = { yes, yes, no, no };

    res[if_statement]      = { yes, no,  no, no };
    res[while_statement]   = { yes, yes, no, no };
    res[if_else_statement] = { yes, yes, no, no };

    res[eq_expr]  = { yes, { false, false, true, false, false, false}, no, no };
    res[neq_expr] = { yes, yes, no, no };
    res[lt_expr]  = { yes, no,  no, no };
    res[gt_expr]  =  {yes, no,  no, no };
    res[lte_expr] = { yes, { false, false, true, false, false, false}, no, no };
    res[gte_expr] = { yes, { false, false, true, false, false, false}, no, no };

    res[func_arg]                   = { yes,  no, no, no };
    res[func_arg_float_as_int]      = { yes,  no, no, no };
    res[func_arg_on_stack]          = { yes, yes, no, no };
    res[func_call_arg]              = { yes,  no, no, no };
    res[func_call_arg_float_as_int] = { yes,  no, no, no };
    res[func_call_arg_on_stack]     = { yes, yes, no, no };

    return res.get();
}

constexpr auto has_instr_mapping = generate_has_instr();

/* NODE_GET_PARENT_ARG_IDX_LOOKUP */

constexpr auto generate_parent_arg_idx_lookup() {
    using enum rv_node_type;

    mapping_helper<std::array<ArrayForTypes<int8_t>, 4>, max_node_types> res;

    ArrayForTypes<int8_t> zero { 0, 0, 0, 0, 0, 0 };
    ArrayForTypes<int8_t> one  { 1, 1, 1, 1, 1, 1 };
    ArrayForTypes<int8_t> two  { 2, 2, 2, 2, 2, 2 };

    res[invalid]        = { two,  zero, zero, zero };
    res[statement_list] = { two,  zero, zero, zero };
    res[empty]          = { zero, zero, zero, zero };
    res[expression]     = { two,  zero, zero, zero };

    res[add_expr]       = { two, zero, zero, zero };
    res[sub_expr]       = { two, zero, zero, zero };
    res[mul_expr]       = { two, zero, zero, zero };
    res[div_expr]       = { two, zero, zero, zero };
    res[mod_expr]       = { two, zero, zero, zero };
    res[bitand_expr]    = { two, zero, zero, zero };
    res[bitor_expr]     = { two, zero, zero, zero };
    res[bitxor_expr]    = { two, zero, zero, zero };
    res[lshift_expr]    = { two, zero, zero, zero };
    res[rshift_expr]    = { two, zero, zero, zero };
    res[urshift_expr]   = { two, zero, zero, zero };
    res[logic_and_expr] = { two, zero, zero, zero };
    res[logic_or_expr]  = { two, zero, zero, zero };

    res[bitnot_expr]    = { two, zero, zero, zero };
    res[logic_not_expr] = { two, zero, zero, zero };
    res[neg_expr]       = { two, zero, zero, zero };

    res[literal_expr]     = { zero, { 1, 1, 1, 0, 1, 1 }, { 0, 0, 0, 1, 0, 0 }, zero };
    res[cast_expr]        = { two, zero, zero, zero };
    res[deref_expr]       = { two, zero, zero, zero };
    res[assign_expr]      = { zero, one, zero, zero };
    res[decl_expr]        = { two, zero, zero, zero };
    res[id_expr]          = { two, zero, zero, zero };
    res[while_dummy]      = { two, zero, zero, zero };
    res[func_decl_dummy]  = { two, zero, zero, zero };
    res[return_statement] = { zero, zero, zero, zero };

    res[func_decl]     = { zero, zero, zero, zero };
    res[func_arg_list] = { two,  zero, zero, zero };

    res[func_call_expression] = { zero, zero, { 0, 0, 1, 1, 0, 0 }, zero};
    res[func_call_arg_list]   = { two,  zero, zero, zero };

    res[if_statement]      = { two, zero, zero, zero };
    res[while_statement]   = { two, zero, zero, zero };
    res[if_else_statement] = { two, zero, zero, zero };

    res[eq_expr]  = { two, zero, zero, zero };
    res[neq_expr] = { two, zero, zero, zero };
    res[lt_expr]  = { two, zero, zero, zero };
    res[gt_expr]  = { two, zero, zero, zero };
    res[lte_expr] = { two, zero, zero, zero };
    res[gte_expr] = { two, zero, zero, zero };

    res[func_arg]                   = { zero, zero, zero, zero };
    res[func_arg_float_as_int]      = { zero, zero, zero, zero };
    res[func_arg_on_stack]          = { zero, zero, zero, zero };
    res[func_call_arg]              = { zero, zero, zero, zero };
    res[func_call_arg_float_as_int] = { zero, zero, zero, zero };
    res[func_call_arg_on_stack]     = { zero, zero, zero, zero };

    return res.get();
}

constexpr auto parent_arg_idx_lookup = generate_parent_arg_idx_lookup();

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


