<?php

namespace Rubix\ML\NeuralNet\Initializers;

use Rubix\ML\Other\Structures\Matrix;

interface Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @param  int  $fanIn
     * @param  int  $fanOut
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : Matrix;
}
