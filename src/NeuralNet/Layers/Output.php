<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Output extends Layer
{
    /**
     * Calculate a backward pass of the network from the output layer.
     *
     * @param  array  $labels
     * @return void
     */
    public function back(array $labels) : void;
}
