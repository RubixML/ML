<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Stringable;

/**
 * Optimizer
 *
 * An optimizer takes in a parameter and its gradient and computes a step tensor
 * that it subtracts from the parameter in place. Every optimizer is paired with
 * a Scheduler that controls the learning rate.
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
     * Compute a step of gradient descent for a single parameter and update the
     * parameter in place.
     *
     * @internal
     *
     * @param Parameter $param
     */
    public function update(Parameter $param) : void;

    /**
     * Flush the parameter cache.
     *
     * @internal
     */
    public function flush() : void;
}
