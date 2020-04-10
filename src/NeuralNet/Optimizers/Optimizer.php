<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;

interface Optimizer
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor;
}
