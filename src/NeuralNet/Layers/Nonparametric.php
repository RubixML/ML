<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;

interface Nonparametric extends Layer
{
    /**
     * Compute the training activations of each neuron in the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix;

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix;
}
