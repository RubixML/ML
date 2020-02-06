<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameters\Parameter;

interface Adaptive extends Optimizer
{
    /**
     * Warm the parameter cache.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     */
    public function warm(Parameter $param) : void;
}
