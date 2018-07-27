<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Parametric extends Layer
{
    const PHI = 1e8;

    /**
     * Initialize the layer with an indegree and optimizer instance.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function initialize(int $fanIn, Optimizer $optimizer) : int;

    /**
     * Update the parameters in the layer and return the magnitude of the step.
     *
     * @return float
     */
    public function update() : float;

    /**
     * Read the parameters and return them in an associative array.
     *
     * @return array
     */
    public function read() : array;

    /**
     * Restore the parameters in the layer.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void;
}
