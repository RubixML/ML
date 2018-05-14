<?php

namespace Rubix\Engine\NeuralNet\Layers;

interface Output extends Layer
{
    /**
     * Calculate a backward pass and return an array of erros and gradients.
     *
     * @param  mixed  $outcome
     * @return array
     */
    public function back($outcome) : array;
}
