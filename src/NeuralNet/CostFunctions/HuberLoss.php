<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Huber Loss
 *
 * The pseudo Huber Loss function transitions between L1 and L2 (Least Squares)
 * loss at a given pivot point (*alpha*) such that the function becomes more
 * quadratic as the loss decreases. The combination of L1 and L2 loss makes
 * Huber Loss robust to outliers while maintaining smoothness near the minimum.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HuberLoss implements RegressionLoss
{
    /**
     * The alpha quantile i.e the pivot point at which numbers larger will be
     * evalutated with an L1 loss while number smaller will be evalutated with
     * an L2 loss.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The square of the alpha parameter.
     *
     * @var float
     */
    protected $alpha2;

    /**
     * @param float $alpha
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $alpha = 0.9)
    {
        if ($alpha <= 0.0) {
            throw new InvalidArgumentException('Alpha must be greater than'
                . " 0, $alpha given.");
        }

        $this->alpha = $alpha;
        $this->alpha2 = $alpha ** 2;
    }

    /**
     * Compute the loss score.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        return $target->subtract($output)->map([$this, '_compute'])->mean()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix
    {
        $alpha = $output->subtract($target);

        return $alpha->square()
            ->add($this->alpha2)
            ->pow(-0.5)
            ->multiply($alpha);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $this->alpha2 * (sqrt(1.0 + ($z / $this->alpha) ** 2) - 1.0);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Huber Loss (alpha: {$this->alpha})";
    }
}
