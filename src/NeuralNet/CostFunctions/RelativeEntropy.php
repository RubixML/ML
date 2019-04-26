<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;

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
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return float
     */
    public function compute(Tensor $expected, Tensor $output) : float
    {
        $expected = $expected->clip(EPSILON, 1.);
        $output = $output->clip(EPSILON, 1.);

        return $expected->divide($output)->log()
            ->multiply($expected)->sum()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return \Rubix\Tensor\Tensor
     */
    public function differentiate(Tensor $expected, Tensor $output) : Tensor
    {
        $expected = $expected->clip(EPSILON, 1.);
        $output = $output->clip(EPSILON, 1.);

        return $output->subtract($expected)
            ->divide($output);
    }
}
