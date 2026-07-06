<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Stochastic
 *
 * SGD (Stochastic Gradient Descent) optimizer -
 * a constant learning rate gradient descent optimizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Stochastic implements Optimizer
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * @param float $rate
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException(
                "Learning rate must be greater than 0, $rate given."
            );
        }

        $this->rate = $rate;

        ExtensionIsLoaded::with('RubixNumPower')->check();
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * SGD update (element-wise):
     *   Δθ_t = η · g_t
     *
     * where:
     *   - g_t is the current gradient,
     *   - η is the learning rate.
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray
    {
        return NumPower::multiply($gradient, $this->rate);
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
        return "Stochastic (rate: {$this->rate})";
    }
}
