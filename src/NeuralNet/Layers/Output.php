<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Output extends Parametric
{
    /**
     * Calculate the errors and gradients for each output neuron.
     *
     * @param  array  $labels
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return array
     */
    public function back(array $labels, Optimizer $optimizer) : array;

    /**
     * Return the activations matrix.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function activations() : Matrix;
}
