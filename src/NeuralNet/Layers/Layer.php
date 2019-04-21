<?php

namespace Rubix\ML\NeuralNet\Layers;

interface Layer
{
    public const EPSILON = 1e-8;

    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @return int
     */
    public function width() : int;

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int;
}
