<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;

interface Input extends Layer
{
    /**
     * Feed the input forward to the next layer in the network.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix;

    /**
     * Forward pass during inference.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix;
}
