<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\Neuron;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

class Output extends Layer
{
    /**
     * @param  int  $n
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($n, ActivationFunction $activationFunction)
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of outputs must be greater than 0.');
        }

        parent::__construct($n);

        for ($i = 0; $i < $n; $i++) {
            $this[$i] = new Neuron($activationFunction);
        }
    }
}
