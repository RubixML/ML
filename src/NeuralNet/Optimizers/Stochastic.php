<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
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
            throw new InvalidArgumentException('The learning rate must'
                . " be greater than 0, $rate given.");
        }

        $this->rate = $rate;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Rubix\Tensor\Matrix $gradient
     */
    public function step(Parameter $param, Matrix $gradient) : void
    {
        $step = $gradient->multiply($this->rate);

        $param->update($step);
    }
}
