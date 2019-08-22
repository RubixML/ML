<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;

interface Optimizer
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Rubix\Tensor\Tensor $gradient
     * @return \Rubix\Tensor\Tensor;
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor;
}
