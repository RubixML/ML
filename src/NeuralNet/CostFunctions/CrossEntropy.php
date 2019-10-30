<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Tensor;
use Tensor\Matrix;

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
     * Compute the loss score.
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        $entropy = $output->clipLower(EPSILON)->log();

        return $target->negate()->multiply($entropy)->mean()->mean();
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
        $tensor = get_class($target);
        
        $denominator = $tensor::ones(...$target->shape())
            ->subtract($output)
            ->multiply($output)
            ->clipLower(EPSILON);

        return $output->subtract($target)
            ->divide($denominator);
    }
}
