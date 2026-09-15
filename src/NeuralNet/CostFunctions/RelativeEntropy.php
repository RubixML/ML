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
     * @param Matrix $z
     * @param Matrix $y
     * @return float
     */
    public function compute(Matrix $z, Matrix $y) : float
    {
        $y = $y->clip(EPSILON, 1.0);
        $z = $z->clip(EPSILON, 1.0);

        return $y->divide($z)->log()
            ->multiply($y)
            ->mean()
            ->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return Matrix
     */
    public function differentiate(Matrix $z, Matrix $y) : Matrix
    {
        $y = $y->clip(EPSILON, 1.0);
        $z = $z->clip(EPSILON, 1.0);

        return $y->negate()->divide($z);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Relative Entropy';
    }
}
