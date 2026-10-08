<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;

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
class LeastSquares implements RegressionLoss
{
    /**
     * Compute the loss score.
     *
     * The loss is the mean squared error over all elements of the matrix, where m
     * is the number of output nodes and n is the number of samples.
     *
     * L(y, ŷ) = Σ(ŷ - y)² / (m * n)
     *
     * @internal
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return float
     */
    public function compute(Matrix $z, Matrix $y) : float
    {
        return $z->subtract($y)->square()->mean()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * The returned gradient is unnormalized. Scaling it by 1 / (m * n) yields the
     * derivative of the loss score returned by compute(). The factor of 2 is
     * required because the loss score is the mean squared error.
     *
     * ∂L/∂ŷ = 2 * (ŷ - y)
     *
     * @internal
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return Matrix
     */
    public function differentiate(Matrix $z, Matrix $y) : Matrix
    {
        return $z->subtract($y)->multiplyScalar(2.0);
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
        return 'Least Squares';
    }
}
