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
     * The underlying learning rate scheduler instance.
     *
     * @internal
     */
    public function scheduler() : Scheduler
    {
        return $this->scheduler;
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
        //
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
     * Take a step of gradient descent for a set of parameters.
     *
     * @internal
     *
     * @param list<array{Parameter, Tensor<int|float|array>}> $gradients
     */
    public function step(array $gradients) : void
    {
        foreach ($gradients as [$param, $gradient]) {
            $param->update($this->update($param, $gradient));
        }

        $this->scheduler->tick();
    }

    /**
     * Flush the parameter cache.
     *
     * @internal
     */
    public function flush() : void
    {
        //
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
