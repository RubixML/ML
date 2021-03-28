<?php

namespace Rubix\ML
{
    /**
     * The current version of the library.
     *
     * @internal
     *
     * @var string
     */
    const VERSION = '0.4.1';

    /**
     * A small number used in substitution of 0.
     *
     * @internal
     *
     * @var float
     */
    const EPSILON = 1e-8;

    /**
     * The natural logarithm of the epsilon constant.
     *
     * @internal
     *
     * @var float
     */
    const LOG_EPSILON = -18.420680744;

    /**
     * The number of radians in a full circle.
     *
     * @internal
     *
     * @var float
     */
    const TWO_PI = 2.0 * M_PI;

    /**
     * Half of pi.
     *
     * @internal
     *
     * @var float
     */
    const HALF_PI = 0.5 * M_PI;

    /**
     * Coefficient that determines floating point precision of random number generation.
     *
     * @internal
     *
     * @var int
     */
    const PHI = 100000000;
}
