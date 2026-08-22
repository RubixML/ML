<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers;

use NumPower;
use NDArray;

/**
 * Xavier 2 Normal
 *
 * The Xavier 2 Normal initializer draws from a truncated normal distribution with
 * mean 0 and standard deviation equal to (2 / (fanIn + fanOut)) ** 0.25. This
 * initializer is best suited for layers that feed into an activation layer that
 * outputs values between -1 and 1 such as Hyperbolic Tangent and Softsign.
 *
 * References:
 * [1] X. Glorot et al. (2010). Understanding the Difficulty of Training Deep
 * Feedforward Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Xavier2Normal extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        // Xavier-2 uses fourth-root scaling instead of standard square-root Xavier 1 scaling.
        $stdDev = (2.0 / ($fanOut + $fanIn)) ** 0.25;

        return NumPower::truncatedNormal(shape: [$fanOut, $fanIn], loc: 0.0, scale: $stdDev);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Xavier-2 Normal';
    }
}
