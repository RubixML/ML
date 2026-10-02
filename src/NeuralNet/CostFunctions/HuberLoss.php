<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;

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
    protected float $alpha;

    /**
     * The square of the alpha parameter.
     *
     * @var float
     */
    protected float $alpha2;

    /**
     * @param float $alpha
     * @throws InvalidArgumentException
     */
    public function __construct(float $alpha = 0.9)
    {
        if (ExtensionIsLoaded::with('tensor')->passes()) {
            ExtensionMinimumVersion::with('tensor', '4.0.0')->check();
        }

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
     * @param Matrix $z
     * @param Matrix $y
     * @return float
     */
    public function compute(Matrix $z, Matrix $y) : float
    {
        return $y->subtract($z)
            ->divideScalar($this->alpha)
            ->square()
            ->addScalar(1.0)
            ->sqrt()
            ->subtractScalar(1.0)
            ->multiplyScalar($this->alpha2)
            ->mean()
            ->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return Matrix
     */
    public function differentiate(Matrix $z, Matrix $y) : Matrix
    {
        $beta = $z->subtract($y);

        return $beta->square()
            ->addScalar($this->alpha2)
            ->sqrt()
            ->reciprocal()
            ->multiply($beta)
            ->multiplyScalar($this->alpha);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Huber Loss (alpha: {$this->alpha})";
    }
}
