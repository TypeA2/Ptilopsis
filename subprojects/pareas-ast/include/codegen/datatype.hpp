#ifndef _PAREAS_CODEGEN_DATATYPE_HPP
#define _PAREAS_CODEGEN_DATATYPE_HPP

#include <iosfwd>
#include <cstdint>

enum class DataType : uint8_t {
    INVALID   /* = 0b000 */,
    VOID      /* = 0b001 */,
    INT       /* = 0b010 */,
    FLOAT     /* = 0b011 */,
    INT_REF   /* = 0b100 */,
    FLOAT_REF /* = 0b101 */
};

DataType reference_of(DataType);
DataType value_of(DataType);

std::ostream& operator<<(std::ostream&, const DataType&);

#endif
