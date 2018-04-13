<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;

class OutputLayer extends Layer
{
    /**
     * @param  array  $outcomes
     * @param  \Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction  $activationFunction
     * @return void
     */
    public function __construct(array $outcomes, ActivationFunction $activationFunction)
    {
        $outcomes = array_values(array_unique($outcomes));

        parent::__construct(count($outcomes));

        foreach ($outcomes as $i => $outcome) {
            $this[$i] = new Output($outcome, $activationFunction);
        }
    }
}
