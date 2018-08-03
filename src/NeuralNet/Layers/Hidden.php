<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Parametric
{
    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevWeights
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevErrors
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return array
     */
    public function back(Matrix $prevWeights, Matrix $prevErrors, Optimizer $optimizer) : array;
}
