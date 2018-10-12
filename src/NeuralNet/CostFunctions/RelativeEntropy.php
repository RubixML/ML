<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

/**
 * Relative Entropy
 *
 * Relative Entropy or *Kullback-Leibler divergence* is a measure of how the
 * expectation and activation of the network diverge.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RelativeEntropy implements CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-INF, INF];
    }

    /**
     * Compute the loss matrix.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        $expected = $expected->clip(self::EPSILON, 1.);
        $activations = $activations->clip(self::EPSILON, 1.);

        return $expected->divide($activations)->log()
            ->multiply($expected);
    }

    /**
     * Calculate the gradient of the cost function with respect to the
     * activation.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @param  \Rubix\Tensor\Matrix  $delta
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        $expected = $expected->clip(self::EPSILON, 1.);
        $activations = $activations->clip(self::EPSILON, 1.);

        return $activations->subtract($expected)
            ->divide($activations);
    }
}
