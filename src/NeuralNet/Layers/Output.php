<?php

namespace Rubix\Engine\NeuralNet\Layers;

interface Output extends Parametric
{
    /**
     * Calculate a backward pass of the network from the output layer.
     *
     * @param  array  $labels
     * @return void
     */
    public function back(array $labels) : void;
}
