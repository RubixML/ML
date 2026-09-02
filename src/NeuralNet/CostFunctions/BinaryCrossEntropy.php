<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\Traits\AssertsShapes;
use const Rubix\ML\EPSILON;

/**
 * Binary Cross Entropy
 *
 * Binary Cross Entropy, or log loss, measures the performance of a binary
 * classification model whose output is a probability value between 0 and 1.
 * Cross-entropy loss increases as the predicted probability diverges from the
 * actual label. So predicting a probability of .012 when the actual observation
 * label is 1 would be bad and result in a high loss value. A perfect score
 * would have a log loss of 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class BinaryCrossEntropy implements ClassificationLoss
{
    use AssertsShapes;

    /**
     * Compute the loss score.
     *
     * L(y, ŷ) = -Σ(y * log(ŷ) + (1 - y) * log(1 - ŷ)) / n
     *
     * @param Matrix $output
     * @param Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        $this->assertSameShape($output, $target);

        $output = $output->clip(EPSILON, 1.0 - EPSILON);
        $target = $target->clip(EPSILON, 1.0 - EPSILON);

        $oneMinusOutput = Matrix::ones(...$output->shape())->subtract($output);
        $oneMinusTarget = Matrix::ones(...$target->shape())->subtract($target);

        return $target
            ->multiply($output->log())
            ->add($oneMinusTarget->multiply($oneMinusOutput->log()))
            ->negate()
            ->mean()
            ->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = (ŷ - y) / (ŷ * (1 - ŷ))
     *
     * @param Matrix $output
     * @param Matrix $target
     * @return Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix
    {
        $this->assertSameShape($output, $target);

        $clippedOutput = $output->clip(EPSILON, 1.0 - EPSILON);

        $oneMinusOutput = Matrix::ones(...$clippedOutput->shape())->subtract($clippedOutput);

        $denominator = $clippedOutput
            ->multiply($oneMinusOutput)
            ->clip(EPSILON, 1.0);

        return $output->subtract($target)->divide($denominator);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Binary Cross Entropy';
    }
}
