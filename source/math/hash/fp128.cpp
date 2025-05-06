#include "hash.impl.h"

using Scalar = fp128;

template std::size_t hash::hash_buffer(const Scalar *v, unsigned long size, std::size_t seed);
