<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use NDArray;
use NumPower;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function get_class;

/**
 * Momentum
 *
 * Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding
 * a factor of the previous velocity to the current step.
 *
 * References:
 * [1] D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.
 * [2] I. Sutskever et al. (2013). On the importance of initialization and momentum in deep learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Momentum implements Optimizer, Adaptive
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The rate at which the momentum force decays.
     *
     * @var float
     */
    protected float $decay;

    /**
     * Should we employ Nesterov's lookahead (NAG) when updating the parameters?
     *
     * @var bool
     */
    protected bool $lookahead;

    /**
     * The parameter cache of velocity NDArrays.
     *
     * @var NDArray[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param float $rate
     * @param float $decay
     * @param bool $lookahead
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $decay = 0.1, bool $lookahead = false)
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
        $this->lookahead = $lookahead;
    }

    /**
     * Warm the cache.
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

        $this->cache[$param->id()] = NumPower::zeros($param->param()->shape());
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * Mathematical formulation (per-parameter element):
     * - Velocity update: v_t = β · v_{t-1} + η · g_t
     *   where β = 1 − decay and η = rate, and g_t is the current gradient.
     * - Returned step (the amount added to the parameter by the trainer): Δθ_t = v_t
     *
     * Nesterov lookahead (when lookahead = true):
     * - We apply the same velocity update a second time to approximate NAG:
     *   v_t ← β · v_t + η · g_t
     *
     * Notes:
     * - This method updates and caches the velocity tensor per Parameter id.
     * - The actual parameter update is performed by the training loop using the returned velocity.
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray
    {
        $velocity = $this->cache[$param->id()];

        // velocity = gradient * rate + velocity * (1 - decay)
        $velocity = NumPower::add(
            NumPower::multiply($gradient, $this->rate),
            NumPower::multiply($velocity, 1.0 - $this->decay)
        );

        $this->cache[$param->id()] = $velocity;

        if ($this->lookahead) {
            // Apply lookahead: velocity = gradient * rate + velocity * (1 - decay)
            $velocity = NumPower::add(
                NumPower::multiply($gradient, $this->rate),
                NumPower::multiply($velocity, 1.0 - $this->decay)
            );
        }

        return $velocity;
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
        return "Momentum (rate: {$this->rate}, decay: {$this->decay},"
            . ' lookahead: ' . Params::toString($this->lookahead) . ')';
    }
}
