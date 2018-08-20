<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Parametric extends Nonparametric
{
    /**
     * Read the parameters and return them in an associative array.
     *
     * @return array
     */
    public function read() : array;

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void;
}
