<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;

interface Adaptive extends Optimizer
{
    /**
     * Initialize a parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $param
     * @return void
     */
    public function initialize(Parameter $param) : void;
}
