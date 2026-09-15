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
     * @param Matrix $z
     * @param Matrix $y
     * @return float
     */
    public function compute(Matrix $z, Matrix $y) : float
    {
        if ($z->shape() !== $y->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        $clippedOutput = $z->clip(EPSILON, 1.0);

        return $y
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
     * @param Matrix $z
     * @param Matrix $y
     * @return Matrix
     */
    public function differentiate(Matrix $z, Matrix $y) : Matrix
    {
        if ($z->shape() !== $y->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        return $y->negate()->divide($z->clip(EPSILON, 1.0));
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
