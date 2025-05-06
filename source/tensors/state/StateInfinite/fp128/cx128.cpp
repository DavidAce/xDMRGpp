#include "../../StateInfinite.tmpl.h"

using Scalar = fp128;
using T = cx128;

template StateInfinite<Scalar>::StateInfinite(const StateInfinite<T> &other) noexcept;
template StateInfinite<Scalar> &StateInfinite<Scalar>::operator=(const StateInfinite<T> &other) noexcept;
