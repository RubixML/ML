<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Parametric extends Layer
{
    /**
     * Update the parameters in the layer and return the magnitude of the step.
     *
     * @return float
     */
    public function update() : float;

    /**
     * Read the parameters and return them in an array.
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
