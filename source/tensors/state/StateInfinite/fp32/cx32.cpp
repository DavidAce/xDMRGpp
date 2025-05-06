#include "../../StateInfinite.tmpl.h"

using Scalar = fp32;
using T = cx32;

template StateInfinite<Scalar>::StateInfinite(const StateInfinite<T> &other) noexcept;
template StateInfinite<Scalar> &StateInfinite<Scalar>::operator=(const StateInfinite<T> &other) noexcept;
