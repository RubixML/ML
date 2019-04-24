<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;

use const Rubix\ML\EPSILON;

/**
 * Cross Entropy
 *
 * Cross Entropy, or log loss, measures the performance of a classification model
 * whose output is a probability value between 0 and 1. Cross-entropy loss
 * increases as the predicted probability diverges from the actual label. So
 * predicting a probability of .012 when the actual observation label is 1 would
 * be bad and result in a high loss value. A perfect score would have a log loss
 * of 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CrossEntropy implements CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., INF];
    }

    /**
     * Compute the matrix.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $output
     * @return float
     */
    public function compute(Matrix $expected, Matrix $output) : float
    {
        return $expected->negate()->multiply($output->log())->sum()->mean();
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
        $denominator = Matrix::ones(...$output->shape())
            ->subtract($output)
            ->multiply($output)
            ->clipLower(EPSILON);

        return $output->subtract($expected)
            ->divide($denominator);
    }
}
