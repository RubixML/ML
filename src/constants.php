<?php

namespace Rubix\ML
{
    /**
     * Major versions include architectural updates and backwards compatibility breaks.
     *
     * @var string
     */
    const MAJOR_VERSION = '0';

    /**
     * Minor versions include new features and deprecations.
     *
     * @var string
     */
    const MINOR_VERSION = '2';

    /**
     * Bugfix versions contain fixes as well as refactorings.
     *
     * @var string
     */
    const BUGFIX_VERSION = '4';

    /**
     * The full version number.
     *
     * @var string
     */
    const VERSION = MAJOR_VERSION . '.' . MINOR_VERSION . '.' . BUGFIX_VERSION;

    /**
     * A small number used in substitution of 0.
     *
     * @var float
     */
    const EPSILON = 1e-8;

    /**
     * The natural logarithm of the epsilon constant.
     *
     * @var float
     */
    const LOG_EPSILON = -18.420680744;

    /**
     * The number of radians in a full circle.
     *
     * @var float
     */
    const TWO_PI = 2.0 * M_PI;

    /**
     * Half of pi.
     */
    const HALF_PI = 0.5 * M_PI;

    /**
     * Coefficient that determines floating point precision of random number generation.
     *
     * @var int
     */
    const PHI = 100000000;
}
