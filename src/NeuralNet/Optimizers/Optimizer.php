<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Layers\Parametric;

interface Optimizer
{
    const EPSILON = 1e-8;

    /**
     * Initialize the optimizer for a particular layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return void
     */
    public function initialize(Parametric $layer) : void;

    /**
     * Calculate the step for a parametric layer and return the magnitude.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Parametric  $layer
     * @return float
     */
    public function step(Parametric $layer) : float;
}
