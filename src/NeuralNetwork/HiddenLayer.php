<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;

class HiddenLayer extends Layer
{
    /**
     * @param  int  $n
     * @param  \Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction  $activationFunction
     * @return void
     */
    public function __construct(int $n, ActivationFunction $activationFunction)
    {
        parent::__construct($n + 1);

        for ($i = 0; $i < $n; $i++) {
            $this[$i] = new Hidden($activationFunction);
        }

        $this[count($this) - 1] = new Bias();
    }
}
