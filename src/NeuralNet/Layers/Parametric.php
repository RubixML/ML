<?php

namespace Rubix\Engine\NeuralNet\Layers;

interface Parametric
{
    /**
     * Generate a random weight for each synapse in the layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Layer
     * @return void
     */
    public function initialize(Layer $previous) : void;

    /**
     * Update the parameters in the layer.
     *
     * @param  array  $steps
     * @return void
     */
    public function update(array $steps) : void;
}
