<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;

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
class CrossEntropy implements ClassificationLoss
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
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return float
     */
    public function compute(Tensor $expected, Tensor $output) : float
    {
        return $expected->negate()->multiply($output->log())->sum()->mean();
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
        $tensor = get_class($expected);
        
        $denominator = $tensor::ones(...$expected->shape())
            ->subtract($output)
            ->multiply($output)
            ->clipLower(EPSILON);

        return $output->subtract($expected)
            ->divide($denominator);
    }
}
