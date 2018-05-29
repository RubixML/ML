<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\Engine\NeuralNet\Layers\Parametric;

interface Optimizer
{
    const EPSILON = 1e-8;

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Network  $network
     * @return void
     */
    public function initialize(Parametric $layer) : void;

    /**
     * Calculate the step for a parametric layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Parametric  $layer
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parametric $layer) : Matrix;
}
