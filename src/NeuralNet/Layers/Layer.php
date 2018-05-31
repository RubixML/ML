<?php

namespace Rubix\Engine\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Layer
{
    const EPSILON = 1e-8;

    /**
     * Compute the input sum and activation of each nueron in the layer and return
     * an activation matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix;
}
