<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Parametric extends Nonparametric
{
    /**
     * Return the parameters of the layer.
     *
     * @return \Rubix\ML\NeuralNet\Parameter[]
     */
    public function parameters() : array;

    /**
     * Return the parameters of the layer in an associative array.
     *
     * @return array
     */
    public function read() : array;

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param array $parameters
     */
    public function restore(array $parameters) : void;
}
