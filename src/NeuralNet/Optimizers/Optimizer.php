<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;

interface Optimizer
{
    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \Rubix\ML\Other\Structures\Matrix  $gradient
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradient) : Matrix;
}
