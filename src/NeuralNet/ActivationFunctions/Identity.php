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
}
