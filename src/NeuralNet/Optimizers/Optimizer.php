<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Stringable;

/**
 * Optimizer
 *
 * An optimizer takes in a parameter and its gradient and returns a step tensor
 * that is subtracted from the parameter by the parameter object. Every
 * optimizer is paired with a Scheduler that controls the learning rate.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Optimizer extends Stringable
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param Tensor<int|float|array> $gradient
     * @return Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor;

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void;

    /**
     * Reset the parameter cache.
     *
     * @internal
     */
    public function reset() : void;

    /**
     * Return the learning-rate scheduler paired with the optimizer.
     *
     * @return Scheduler
     */
    public function scheduler() : Scheduler;
}
