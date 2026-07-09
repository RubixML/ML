<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Initializers;

use NumPower;
use NDArray;

/**
 * Xavier 1 Uniform
 *
 * The Xavier 2 initializer draws from a uniform distribution [-limit, limit]
 * where *limit* is equal to (6 / ($fanIn + $fanOut)) ** 0.25. This initializer
 * is best suited for layers that feed into an activation layer that outputs
 * values between -1 and 1 such as Hyperbolic Tangent and Softsign.
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
class Xavier2Uniform extends AbstractInitializer
{
    /**
     * @inheritdoc
     */
    public function initialize(int $fanIn, int $fanOut) : NDArray
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);

        // Xavier-2 uses fourth-root scaling instead of standard square-root Xavier 1 scaling.
        $limit = (6.0 / ($fanOut + $fanIn)) ** 0.25;

        return NumPower::uniform(shape: [$fanOut, $fanIn], low: -$limit, high: $limit);
    }

    /**
     * Return the string representation of the initializer.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Xavier-2 Uniform';
    }
}
