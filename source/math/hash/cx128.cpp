#include "hash.impl.h"

using Scalar = cx128;

template std::size_t hash::hash_buffer(const Scalar *v, unsigned long size, std::size_t seed);
