<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;

interface Optimizer
{
    public const EPSILON = 1e-8;
    
    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Rubix\Tensor\Matrix $gradient
     * @return \Rubix\Tensor\Matrix
     */
    public function step(Parameter $param, Matrix $gradient) : Matrix;
}
