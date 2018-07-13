<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;

interface Hidden extends Parametric
{
    /**
     * Calculate the errors and gradients of the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevWeights
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevErrors
     * @return array
     */
    public function back(Matrix $prevWeights, Matrix $prevErrors) : array;
}
