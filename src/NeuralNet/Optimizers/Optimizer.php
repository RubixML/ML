<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;

interface Optimizer
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Tensor\Tensor $gradient
     * @return \Tensor\Tensor;
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor;
}
