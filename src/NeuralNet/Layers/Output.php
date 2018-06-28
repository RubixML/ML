<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Output extends Parametric
{
    /**
     * Initialize the layer.
     *
     * @param  int  $width
     * @return void
     */
    public function initialize(int $width) : void;

    /**
     * Calculate a backward pass of the network from the output layer.
     *
     * @param  array  $labels
     * @return array
     */
    public function back(array $labels) : array;

    /**
     * Return the computed activation matrix.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function activations() : Matrix;
}
