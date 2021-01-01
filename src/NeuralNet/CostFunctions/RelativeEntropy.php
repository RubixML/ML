<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

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
     * Compute the loss.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        $target = $target->clip(EPSILON, 1.0);
        $output = $output->clip(EPSILON, 1.0);

        return $target->divide($output)->log()
            ->multiply($target)
            ->mean()
            ->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix
    {
        $target = $target->clip(EPSILON, 1.0);
        $output = $output->clip(EPSILON, 1.0);

        return $output->subtract($target)
            ->divide($output);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Relative Entropy';
    }
}
