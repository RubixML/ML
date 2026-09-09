<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;

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
     * The learning rate schedule.
     *
     * @var Scheduler
     */
    protected Scheduler $scheduler;

    /**
     * @param Scheduler $scheduler
     */
    public function __construct(Scheduler $scheduler)
    {
        $this->scheduler = $scheduler;
    }

    /**
     * Advance the paired learning-rate schedule by one batch.
     *
     * @internal
     */
    public function step() : void
    {
        $this->scheduler->tick();
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param Tensor<int|float|array> $gradient
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param, Tensor $gradient) : Tensor
    {
        return $gradient->multiply($this->scheduler->rate());
    }

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void
    {
    }

    /**
     * Reset the parameter cache.
     *
     * @internal
     */
    public function reset() : void
    {
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
        return "Stochastic (scheduler: {$this->scheduler})";
    }
}
