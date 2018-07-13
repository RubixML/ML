<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Parametric
{
    /**
     * Initialize the layer.
     *
     * @param  int  $prevWidth
     * @return int
     */
    public function initialize(int $prevWidth, Optimizer $optimizer) : int;

    /**
     * Calculate the errors and gradients of the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevWeights
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevErrors
     * @return array
     */
    public function back(Matrix $prevWeights, Matrix $prevErrors) : array;
}
