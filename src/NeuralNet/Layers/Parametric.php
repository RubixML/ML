<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Parametric extends Layer
{
    const PHI = 1e8;

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int;

    /**
     * Read the parameters and return them in an associative array.
     *
     * @return array
     */
    public function read() : array;

    /**
     * Restore the parameters in the layer.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void;
}
