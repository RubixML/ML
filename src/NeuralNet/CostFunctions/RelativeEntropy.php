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
     * The loss is the mean relative entropy over all elements of the matrix, where
     * m is the number of classes and n is the number of samples.
     *
     * L(y, ŷ) = Σ(y * log(y / ŷ)) / (m * n)
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
     * The returned gradient is unnormalized. Scaling it by 1 / (m * n) yields the
     * derivative of the loss score returned by compute().
     *
     * ∂L/∂ŷ = -y / ŷ
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
