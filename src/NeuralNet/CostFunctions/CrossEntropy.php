<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;

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
     * Compute the loss matrix.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $activations
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        return $activations->negate()
            ->multiply($expected)
            ->add($activations->exp()->add(1.)->log());
    }

    /**
     * Calculate the gradient of the cost function with respect to the
     * activation.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $activations
     * @param \Rubix\Tensor\Matrix $delta
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        $denominator = Matrix::ones(...$activations->shape())
            ->subtract($activations)
            ->multiply($activations);

        return $activations->subtract($expected)
            ->divide($denominator->clipLower(self::EPSILON));
    }
}
