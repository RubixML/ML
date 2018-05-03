<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\Bias;
use Rubix\Engine\NeuralNet\Neuron;
use Rubix\Engine\NeuralNet\ActivationFunctions\ELU;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

class Hidden extends Layer
{
    /**
     * @param  int  $n
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction|null  $activationFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $n, ActivationFunction $activationFunction = null)
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of neurons must be greater than 0.');
        }

        if (!isset($activationFunction)) {
            $activationFunction = new ELU();
        }

        parent::__construct($n + 1);

        for ($i = 0; $i < $n; $i++) {
            $this[$i] = new Neuron($activationFunction);
        }

        $this[count($this) - 1] = new Bias();
    }
}
