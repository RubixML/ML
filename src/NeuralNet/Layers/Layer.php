<?php

namespace Rubix\Engine\NeuralNet\Layers;

interface Layer
{
    /**
     * Compute the weighted sum of the inputs and return an output matrix containing
     * the activation of each neuron for each sample.
     *
     * @param  array  $input
     * @return array
     */
    public function forward(array $input) : array;
}
