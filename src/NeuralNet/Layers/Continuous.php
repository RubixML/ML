<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\Identity;

class Continuous extends Output
{
    /**
     * @return void
     */
    public function __construct()
    {
        parent::__construct(1, new Identity());
    }
}
