<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
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
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     */
    public function warm(Parameter $param) : void;

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param Tensor<int|float|array> $gradient
     * @return Tensor<int|float|array>
     */
    public function update(Parameter $param, Tensor $gradient) : Tensor;

    /**
     * Advance the paired learning-rate schedule by one batch.
     *
     * @internal
     */
    public function step() : void;

    /**
     * Reset the parameter cache.
     *
     * @internal
     */
    public function reset() : void;
}
