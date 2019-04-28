<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use InvalidArgumentException;

/**
 * Stochastic
 *
 * A constant learning rate gradient descent optimizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Stochastic implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * @param float $rate
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.001)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must'
                . " be greater than 0, $rate given.");
        }

        $this->rate = $rate;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Rubix\Tensor\Tensor $gradient
     */
    public function step(Parameter $param, Tensor $gradient) : void
    {
        $param->update($gradient->multiply($this->rate));
    }
}
