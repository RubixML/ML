<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

class PReLU implements ActivationFunction
{
    /**
     * The amount of leakage as a ratio of the input value to allow to pass
     * through when not activated.
     *
     * @var float
     */
    protected $leakage;

    /**
     * @param  float  $leakage
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $leakage = 0.01)
    {
        if ($leakage < 0 or $leakage > 1) {
            throw new InvalidArgumentException('Leakage coefficient must be'
                . ' between 0 and 1.');
        }

        $this->leakage = $leakage;
    }

    /**
     * Compute the output value.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $z
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map(function ($value) {
            return $value >= 0.0 ? $value : $this->leakage * $value;
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
            return $output >= 0.0 ? 1.0 : $this->leakage;
        });
    }

    /**
     * Generate an initial synapse weight range.
     *
     * @param  int  $in
     * @return float
     */
    public function initialize(int $in) : float
    {
        $r = pow(6 / $in, 1 / self::ROOT_2);

        return random_int(-$r * 1e8, $r * 1e8) / 1e8;
    }
}
