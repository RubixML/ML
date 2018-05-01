<?php

namespace Rubix\Engine\NeuralNet;

use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;

class OutputLayer extends Layer
{
    /**
     * @param  array  $outcomes
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction|null  $activationFunction
     * @return void
     */
    public function __construct(array $outcomes, ActivationFunction $activationFunction = null)
    {
        $outcomes = array_values(array_unique($outcomes));
        $n = count($outcomes);

        if (!isset($activationFunction)) {
            $activationFunction = new Sigmoid();
        }

        parent::__construct($n);

        foreach ($outcomes as $i => $outcome) {
            $this[$i] = new Output($outcome, $activationFunction);
        }
    }
}
