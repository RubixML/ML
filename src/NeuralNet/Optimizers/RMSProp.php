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
 * RMS Prop
 *
 * An adaptive gradient technique that divides the current gradient over a rolling window
 * of magnitudes of recent gradients.
 *
 * References:
 * [1] T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the
 * gradient by a running average of its recent magnitude.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class RMSProp implements Optimizer, Adaptive
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The rms decay rate.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The opposite of the rms decay rate.
     *
     * @var float
     */
    protected float $rho;

    /**
     * The cache of running squared gradients.
     *
     * @var NDArray[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param float $rate
     * @param float $decay
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $decay = 0.1)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException(
                "Learning rate must be greater than 0, $rate given."
            );
        }

        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException(
                "Decay must be between 0 and 1, $decay given."
            );
        }

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->rate = $rate;
        $this->decay = $decay;
        $this->rho = 1.0 - $decay;
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
     * RMSProp update (element-wise):
     *   v_t = ρ · v_{t-1} + (1 − ρ) · g_t^2
     *   Δθ_t = η · g_t / max(sqrt(v_t), ε)
     *
     * where:
     *   - g_t is the current gradient,
     *   - v_t is the running average of squared gradients,
     *   - ρ = 1 − decay, η is the learning rate,
     *   - ε is a small constant to avoid division by zero (implemented by clipping √v_t to [ε, +∞)).
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

        $norm = NumPower::add(
            NumPower::multiply($norm, $this->rho),
            NumPower::multiply(NumPower::square($gradient), $this->decay)
        );

        $this->cache[$param->id()] = $norm;

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
        return "RMS Prop (rate: {$this->rate}, decay: {$this->decay})";
    }
}
