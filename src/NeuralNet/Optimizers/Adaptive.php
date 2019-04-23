<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;

interface Adaptive extends Optimizer
{
    /**
     * Warm the cache with a parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     */
    public function warm(Parameter $param) : void;
}
