<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Stringable;

/**
 * Optimizer
 *
 * An optimizer takes in a parameter and its gradient and computes a step tensor
 * that is subtracted from the parameter by the optimizer. Every optimizer is
 * paired with a Scheduler that controls the learning rate.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Optimizer extends Stringable
{
    /**
     * Return the underlying learning rate scheduler instance.
     *
     * @internal
     */
    public function scheduler() : Scheduler;

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void;

    /**
     * Compute a step of gradient descent for a single parameter. The step is
     * not applied here, it is applied by the caller.
     *
     * @internal
     *
     * @param Parameter $param
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param) : Tensor;

    /**
     * Flush the parameter cache.
     *
     * @internal
     */
    public function flush() : void;
}
