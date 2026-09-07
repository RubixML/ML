<?php

namespace Rubix\ML\NeuralNet\Initializers;

/**
 * Xavier 2
 *
 * The Xavier 2 initializer is a backward-compatible alias of He. Like He, it
 * draws from a uniform distribution with limits of +/- sqrt(6 / fanIn). It is
 * kept to preserve the name for existing configurations, particularly for
 * layers that feed into an activation layer that outputs values between -1
 * and 1 such as Hyperbolic Tangent and Softsign.
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
