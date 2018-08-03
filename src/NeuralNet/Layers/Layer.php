<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Layer
{
    const EPSILON = 1e-8;

    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @return int
     */
    public function width() : int;
}
