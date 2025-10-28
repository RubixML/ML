<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\CostFunctions\Base\Contracts\RegressionLoss;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss\Exceptions\InvalidAlphaException;
use Rubix\ML\Traits\AssertsShapes;

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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class HuberLoss implements RegressionLoss
{
    use AssertsShapes;

    /**
     * The alpha quantile i.e the pivot point at which numbers larger will be
     * evalutated with an L1 loss while number smaller will be evalutated with
     * an L2 loss.
     *
     * @var float
     */
    protected float $alpha;

    /**
     * The square of the alpha parameter.
     *
     * @var float
     */
    protected float $alpha2;

    /**
     * @param float $alpha
     * @throws InvalidAlphaException
     */
    public function __construct(float $alpha = 0.9)
    {
        if ($alpha <= 0.0) {
            throw new InvalidAlphaException('Alpha must be greater than 0, ' . $alpha . ' given.');
        }

        $this->alpha = $alpha;
        $this->alpha2 = $alpha ** 2;
    }

    /**
     * Compute the loss score.
     *
     * L(y, ŷ) = α²(√(1 + ((y - ŷ)/α)²) - 1)
     *
     * @internal
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
        $this->assertSameShape($output, $target);

        $difference = NumPower::subtract($target, $output);
        $scaled = NumPower::divide($difference, $this->alpha);
        $squared = NumPower::pow($scaled, 2);
        $sqrt = NumPower::sqrt(NumPower::add($squared, 1.0));
        $loss = NumPower::multiply($this->alpha2, NumPower::subtract($sqrt, 1.0));

        return NumPower::mean($loss);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = (ŷ - y) / √(α² + (ŷ - y)²)
     *
     * @internal
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray
    {
        $this->assertSameShape($output, $target);

        $difference = NumPower::subtract($output, $target);
        $squared = NumPower::pow($difference, 2);
        $denominator = NumPower::sqrt(NumPower::add($squared, $this->alpha2));

        return NumPower::divide($difference, $denominator);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Huber Loss (alpha: ' . $this->alpha . ')';
    }
}
