<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function get_class;

use const Rubix\ML\EPSILON;
use const PHP_FLOAT_MAX;

/**
 * AdaGrad
 *
 * Short for Adaptive Gradient, the AdaGrad Optimizer speeds up the learning of
 * parameters that do not change often and slows down the learning of parameters
 * that do enjoy heavy activity.
 *
 * References:
 * [1] J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning
 * and Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class AdaGrad implements Optimizer, Adaptive
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The cache of sum of squared gradients.
     *
     * @var NDArray[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param float $rate
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException("Learning rate must be greater than 0, $rate given.");
        }

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->rate = $rate;
    }

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     * @throws RuntimeException
     */
    public function warm(Parameter $param) : void
    {
        $class = get_class($param->param());

        if (!$class) {
            throw new RuntimeException('Could not locate parameter class.');
        }

        $zeros = NumPower::zeros($param->param()->shape(), $param->param()->dataType(), 0);

        $this->cache[$param->id()] = $zeros;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * AdaGrad update (element-wise):
     *   n_t = n_{t-1} + g_t^2
     *   Δθ_t = η · g_t / max(√n_t, ε)
     *
     * where:
     *   - g_t is the current gradient,
     *   - n_t is the accumulated (running) sum of squared gradients,
     *   - η is the learning rate (rate),
     *   - ε is a small constant to avoid division by zero (implemented via clipping √n_t to [ε, +∞)).
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray
    {
        $norm = $this->cache[$param->id()];

        $norm = NumPower::add($norm, NumPower::square($gradient));

        $this->cache[$param->id()] = $norm;
)
        $denominator = NumPower::sqrt($norm);
        $denominator = NumPower::clip($denominator, EPSILON, PHP_FLOAT_MAX);

        return NumPower::divide(
            NumPower::multiply($gradient, $this->rate),
            $denominator
        );
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
        return "AdaGrad (rate: {$this->rate})";
    }
}
