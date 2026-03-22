#pragma once

#include "config/enum_utils.h"

/*! \brief Controls how saturation is checked.
 *  Given a series
 *      x = [x0,x1,x2...xN],
 *  we generate a new sequence
 *      y = [f0, f1, f2... fN],
 *  where fk = function(xk,...,xN).
 *  That is, we apply a function f on
 *      x,
 *      x with first element removed,
 *      x with first and second elements removed,
 *  and so on.
 *  The function f is either
 *       val: the standard deviation on xk...xN
 *       mov: the standard deviation on the moving average of xk...xN
 *       min: the standard deviation on the minimum of xk...xN
 *       max: the standard deviation on the maximum of xk...xN
 *       mid: the standard deviation on the midpoint between min and max
 */

enum class SaturationPolicy {
    val = 0,     /*!< Check the standard deviation on xk...xN */
    avg = 1,     /*!< Check the standard deviation of the running average of xk...xN */
    med = 2,     /*!< Check the standard deviation of the running median of xk...xN */
    mov = 4,     /*!< Check the standard deviation on the moving average of xk...xN */
    min = 8,     /*!< Check the standard deviation on the minimum of xk...xN */
    max = 16,    /*!< Check the standard deviation on the maximum of xk...xN */
    mid = 32,    /*!< Check the standard deviation on the midpoint between min and max */
    dif = 64,    /*!< Check the average difference between adjacent midpoint samples */
    log = 128,   /*!< Transform x -> log(x) first */
    allow_bitops /*!< Internal sentinel that marks this enum as a bitflag */
};

template<> std::string_view enum2sv(SaturationPolicy item) noexcept;
template<> SaturationPolicy sv2enum<SaturationPolicy>(std::string_view item);
template<> std::string      flag2str(const SaturationPolicy &item) noexcept;
