<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Layers\Parametric;

interface Optimizer
{
    const EPSILON = 1e-8;

    /**
     * Initialize the layer optimizer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function initialize(Matrix $weights) : void;

    /**
     * Calculate a gradient descent step for a layer given a matrix of gradients.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Matrix $gradients) : Matrix;
}
