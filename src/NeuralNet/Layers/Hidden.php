<?php

namespace Rubix\Engine\NeuralNet\Layers;

interface Hidden extends Parametric
{
    /**
     * Calculate the errors and gradients of the layer for each neuron.
     *
     * @param  \Rubix\Engine\NerualNet\Layers\Layer  $next
     * @return void
     */
    public function back(Layer $next) : void;
}
