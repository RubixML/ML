<?php

namespace Rubix\ML\NeuralNet\Initializers;

use function Rubix\ML\warn_deprecated;

/**
 * Xavier 2
 *
 * The Xavier 2 initializer is a backward-compatible alias of He. Like He, it
 * draws from a uniform distribution with limits of +/- sqrt(6 / fanIn). It is
 * kept to preserve the name for existing configurations.
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing
 * Human-Level Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Xavier2 extends He
{
    public function __construct()
    {
        warn_deprecated('The Xavier2 initializer is deprecated, use the He initializer instead.');
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Xavier 2';
    }
}
