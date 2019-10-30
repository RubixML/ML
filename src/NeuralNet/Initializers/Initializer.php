<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Tensor\Matrix;

interface Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @param int $fanIn
     * @param int $fanOut
     * @return \Tensor\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix;
}
