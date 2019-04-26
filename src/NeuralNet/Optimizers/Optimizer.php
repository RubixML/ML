<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;

interface Optimizer
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Rubix\Tensor\Tensor $gradient
     */
    public function step(Parameter $param, Tensor $gradient) : void;
}
