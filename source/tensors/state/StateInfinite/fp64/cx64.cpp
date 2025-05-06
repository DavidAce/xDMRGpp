#include "../../StateInfinite.tmpl.h"

using Scalar = fp64;
using T = cx64;

template StateInfinite<Scalar>::StateInfinite(const StateInfinite<T> &other) noexcept;
template StateInfinite<Scalar> &StateInfinite<Scalar>::operator=(const StateInfinite<T> &other) noexcept;
