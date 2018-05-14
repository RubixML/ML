<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

interface Optimizer
{
    const EPSILON = 1e-8;

    /**
     * Calculate the value of a single step of gradient descent for a given
     * parameter.
     *
     * @param  array  $gradients
     * @return array
     */
    public function step(array $gradients) : array;
}
