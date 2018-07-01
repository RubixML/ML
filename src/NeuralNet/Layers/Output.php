<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Output extends Parametric
{
    /**
     * Initialize the layer.
     *
     * @param  int  $prevWidth
     * @return void
     */
    public function initialize(int $prevWidth) : void;

    /**
     * Calculate the errors and gradients for each output neuron.
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
