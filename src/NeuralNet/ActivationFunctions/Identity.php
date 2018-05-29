<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;

class Identity implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z;
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @param  \MathPHP\LinearAlgebra\Matrix  $computed
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return MatrixFactory::one($computed->getM(), $computed->getN());
    }

    /**
     * Generate an initial synapse weight range based on the indegree of a
     * single neuron. i.e. the number of inputs it has.
     *
     * @param  int  $inDegree
     * @return float
     */
    public function initialize(int $in) : float
    {
        $scale = pow(10, 10);

        return random_int(-3 * $scale, 3 * $scale) / $scale;
    }
}
