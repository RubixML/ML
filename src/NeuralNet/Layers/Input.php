<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Other\Structures\Matrix;

interface Input extends Layer
{
    /**
     * Feed the input forward to the next layer in the network.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function forward(Matrix $input) : Matrix;

    /**
     * Forward pass during inference.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function infer(Matrix $input) : Matrix;
}
