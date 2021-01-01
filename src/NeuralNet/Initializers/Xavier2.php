<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

/**
 * Xavier 1
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
 */
class Xavier2 implements Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @internal
     *
     * @param int $fanIn
     * @param int $fanOut
     * @return \Tensor\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix
    {
        return Matrix::uniform($fanOut, $fanIn)
            ->multiply((6.0 / ($fanOut + $fanIn)) ** 0.25);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Xavier 2';
    }
}
