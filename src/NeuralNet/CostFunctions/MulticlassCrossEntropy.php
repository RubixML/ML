<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * Multiclass Cross Entropy
 *
 * Multiclass Cross Entropy measures the performance of a multiclass
 * classification model whose output is a probability distribution over the
 * possible classes. Cross-entropy loss increases as the predicted probability
 * distribution diverges from the actual distribution.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MulticlassCrossEntropy implements ClassificationLoss
{
    /**
     * Compute the loss score.
     *
     * L(y, ŷ) = -Σ(y * log(ŷ)) / n
     *
     * @param Matrix $output
     * @param Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        $clippedOutput = $output->clip(EPSILON, 1.0);

        return $target
            ->multiply($clippedOutput->log())
            ->negate()
            ->mean()
            ->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = -y / ŷ
     *
     * @param Matrix $output
     * @param Matrix $target
     * @return Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        return $target->negate()->divide($output->clip(EPSILON, 1.0));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Multiclass Cross Entropy';
    }
}
