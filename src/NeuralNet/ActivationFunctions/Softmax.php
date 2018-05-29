<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;

class Softmax implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        $outputs = $cache = [[]];

        foreach ($z->asVectors() as $i => $vector) {
            for ($j = 0; $j < $vector->getN(); $j++) {
                $cache[$i][$j] = exp($vector[$j]);
            }

            $sigma = array_sum($cache[$i]);

            for ($j = 0; $j < $vector->getN(); $j++) {
                $outputs[$j][$i] = $cache[$i][$j] / ($sigma + self::EPSILON);
            }
        }

        return new Matrix($outputs);
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
        return $computed->hadamardProduct(MatrixFactory::one($computed->getM(),
            $computed->getN())->subtract($computed));
    }

    /**
     * Generate an initial synapse weight range.
     *
     * @param  int  $in
     * @return float
     */
    public function initialize(int $in) : float
    {
        $r = sqrt(6 / $in);

        $scale = pow(10, 10);

        return random_int(-$r * $scale, $r * $scale) / $scale;
    }
}
