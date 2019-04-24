<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;

use const Rubix\ML\EPSILON;

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
     * Compute the loss.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $output
     * @return float
     */
    public function compute(Matrix $expected, Matrix $output) : float
    {
        $expected = $expected->clip(EPSILON, 1.);
        $output = $output->clip(EPSILON, 1.);

        return $expected->divide($output)->log()
            ->multiply($expected)->sum()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $output
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $output) : Matrix
    {
        $expected = $expected->clip(EPSILON, 1.);
        $output = $output->clip(EPSILON, 1.);

        return $output->subtract($expected)
            ->divide($output);
    }
}
