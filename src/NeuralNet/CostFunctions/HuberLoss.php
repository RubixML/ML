<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;
use InvalidArgumentException;

/**
 * Huber Loss
 *
 * The pseudo Huber Loss function transitions between L1 and L2 (Least Squares)
 * loss at a given pivot point (*delta*) such that the function becomes more
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
     * The pivot point i.e the point where numbers larger will be evalutated
     * with an L1 loss while number smaller will be evalutated with an L2 loss.
     *
     * @var float
     */
    protected $delta;

    /**
     * The square of delta.
     *
     * @var float
     */
    protected $delta2;

    /**
     * @param float $delta
     * @throws \InvalidArgumentException
     */
    public function __construct(float $delta = 1.)
    {
        if ($delta <= 0.) {
            throw new InvalidArgumentException('Delta must be greater than'
                . " 0, $delta given.");
        }

        $this->delta = $delta;
        $this->delta2 = $delta ** 2;
    }

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
     * Compute the loss.
     *
     * @param \Rubix\Tensor\Tensor $output
     * @param \Rubix\Tensor\Tensor $target
     * @return \Rubix\Tensor\Tensor
     */
    public function compute(Tensor $output, Tensor $target) : Tensor
    {
        return $target->subtract($output)->map([$this, '_compute']);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Tensor $output
     * @param \Rubix\Tensor\Tensor $target
     * @return \Rubix\Tensor\Tensor
     */
    public function differentiate(Tensor $output, Tensor $target) : Tensor
    {
        $alpha = $output->subtract($target);

        return $alpha->square()
            ->add($this->delta2)
            ->pow(-0.5)
            ->multiply($alpha);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $this->delta2 * (sqrt(1. + ($z / $this->delta) ** 2) - 1.);
    }
}
