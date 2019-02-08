<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;

interface Input extends Layer
{
    /**
     * Feed the input forward to the next layer in the network.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix;

    /**
     * Forward pass during inference.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix;
}
