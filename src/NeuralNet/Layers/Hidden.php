<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Hidden extends Parametric
{
    /**
     * Initialize the layer.
     *
     * @param  int  $width
     * @return int
     */
    public function initialize(int $width) : int;

    /**
     * Calculate the errors and gradients of the layer for each neuron.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @param  \MathPHP\LinearAlgebra\Matrix  $errors
     * @return array
     */
    public function back(Matrix $weights, Matrix $errors) : array;
}
