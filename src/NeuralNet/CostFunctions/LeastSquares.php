<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;

/**
 * Least Squares
 *
 * Least Squares or *quadratic* loss is a function that measures the squared
 * error between the target output and the actual output of a network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeastSquares implements CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0., INF];
    }

    /**
     * Compute the cost.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        return $activations->subtract($expected)->square()
            ->multiplyScalar(0.5);
    }

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @param  \Rubix\ML\Other\Structures\Matrix  $delta
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        return $activations->subtract($expected);
    }
}
