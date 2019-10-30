<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Tensor;
use Tensor\Matrix;

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
class RelativeEntropy implements ClassificationLoss
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
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        $target = $target->clip(EPSILON, 1.);
        $output = $output->clip(EPSILON, 1.);

        return $target->divide($output)->log()
            ->multiply($target)
            ->mean()
            ->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Tensor\Tensor $output
     * @param \Tensor\Tensor $target
     * @return \Tensor\Tensor
     */
    public function differentiate(Tensor $output, Tensor $target) : Tensor
    {
        $target = $target->clip(EPSILON, 1.);
        $output = $output->clip(EPSILON, 1.);

        return $output->subtract($target)
            ->divide($output);
    }
}
