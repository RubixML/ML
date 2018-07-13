<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Output extends Parametric
{
    /**
     * Initialize the layer with an indegree and optimizer instance.
     *
     * @param  int  $prevWidth
     * @return int
     */
    public function initialize(int $prevWidth, Optimizer $optimizer) : int;

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
