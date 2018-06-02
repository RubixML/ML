<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;

class SoftPlus implements Rectifier
{
    /**
     * Compute the output value.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map(function ($value) {
            return log(1 + exp($value));
        });
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
        return $computed->map(function ($output) {
            return 1 / (1 + exp(-$output));
        });
    }
}
