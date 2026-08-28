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
 * Adam
 *
 * Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both
 * Momentum and RMS prop to achieve a balance of velocity and stability. In
 * addition to storing an exponentially decaying average of past squared
 * gradients like RMSprop, Adam also keeps an exponentially decaying average
 * of past gradients, similar to Momentum. Whereas Momentum can be seen as a
 * ball running down a slope, Adam behaves like a heavy ball with friction.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Adam implements Optimizer, Adaptive
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The momentum decay rate.
     *
     * @var float
     */
    protected float $momentumDecay;

    /**
     * The decay rate of the previous norms.
     *
     * @var float
     */
    protected float $normDecay;

    /**
     * The parameter cache of running velocity and squared gradients.
     *
     * @var array{0: NDArray, 1: NDArray}[]
     */
    protected array $cache = [
        // id => [velocity, norm]
    ];

    /**
     * @param float $rate
     * @param float $momentumDecay
     * @param float $normDecay
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException(
                "Learning rate must be greater than 0, $rate given."
            );
        }

        if ($momentumDecay <= 0.0 or $momentumDecay >= 1.0) {
            throw new InvalidArgumentException(
                "Momentum decay must be between 0 and 1, $momentumDecay given."
            );
        }

        if ($normDecay <= 0.0 or $normDecay >= 1.0) {
            throw new InvalidArgumentException(
                "Norm decay must be between 0 and 1, $normDecay given."
            );
        }

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->normDecay = $normDecay;
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

        /** @var NDArray $zeros */
        $zeros = NumPower::zeros($param->param()->shape(), 'float32', 0);

        $this->cache[$param->id()] = [clone $zeros, $zeros];
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * Adam update (element-wise):
     *   v_t = v_{t-1} + β1 · (g_t − v_{t-1})        // exponential moving average of gradients
     *   n_t = n_{t-1} + β2 · (g_t^2 − n_{t-1})      // exponential moving average of squared gradients
     *   Δθ_t = η · v_t / max(√n_t, ε)
     *
     * where:
     *   - g_t is the current gradient,
     *   - v_t is the running average of gradients ("velocity"), β1 = momentumDecay,
     *   - n_t is the running average of squared gradients ("norm"), β2 = normDecay,
     *   - η is the learning rate (rate), ε is a small constant to avoid division by zero (implemented by clipping √n_t to [ε, +∞)).
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $vHat = NumPower::multiply(
            NumPower::subtract($gradient, $velocity),
            $this->momentumDecay
        );

        $velocity = NumPower::add($velocity, $vHat);

        $nHat = NumPower::multiply(
            NumPower::subtract(NumPower::square($gradient), $norm),
            $this->normDecay
        );

        $norm = NumPower::add($norm, $nHat);

        $this->cache[$param->id()] = [$velocity, $norm];

        $denominator = NumPower::sqrt($norm);
        $denominator = NumPower::clip($denominator, EPSILON, PHP_FLOAT_MAX);

        return NumPower::divide(
            NumPower::multiply($velocity, $this->rate),
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
        return "Adam (rate: {$this->rate}, momentum decay: {$this->momentumDecay},"
            . " norm decay: {$this->normDecay})";
    }
}
