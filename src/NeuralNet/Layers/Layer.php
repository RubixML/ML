<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Layer
{
    const ROOT_2 = 1.41421356237;

    const EPSILON = 1e-8;

    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @return int
     */
    public function width() : int;

    /**
     * Compute the input sum and activation of each nueron in the layer and return
     * an activation matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix;
}
