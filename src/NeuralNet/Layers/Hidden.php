<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Hidden extends Layer
{
    /**
     * Calculate the errors and gradients of the layer for each neuron.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Layer  $next
     * @return void
     */
    public function back(Layer $next) : void;
}
