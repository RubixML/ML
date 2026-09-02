<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Stringable;

interface Layer extends Stringable
{
    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @internal
     *
     * @return positive-int
     */
    public function width() : int;

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @internal
     *
     * @param positive-int $fanIn
     * @return positive-int
     */
    public function initialize(int $fanIn) : int;

    /**
     * Feed the input forward to the next layer in the network.
     *
     * @internal
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function forward(Matrix $input) : Matrix;

    /**
     * Forward pass during inference.
     *
     * @internal
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function infer(Matrix $input) : Matrix;
}
