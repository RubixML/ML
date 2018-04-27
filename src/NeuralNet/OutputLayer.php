<?php

namespace Rubix\Engine\NeuralNet;

use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;

class OutputLayer extends Layer
{
    /**
     * @param  array  $outcomes
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
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
