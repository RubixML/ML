<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

class PReLU implements Rectifier
{
    /**
     * The threshold at which the neuron activates.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The amount of leakage as a ratio of the input value to allow to pass
     * through when not activated.
     *
     * @var float
     */
    protected $leakage;

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [-INF, INF];
    }

    /**
     * @param  float  $threshold
     * @param  float  $leakage
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $threshold = 0.0, float $leakage = 0.01)
    {
        if ($leakage < 0 or $leakage > 1) {
            throw new InvalidArgumentException('Leakage coefficient must be'
                . ' between 0 and 1.');
        }

        $this->threshold = $threshold;
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
            return $value >= $this->threshold ? $value : $this->leakage * $value;
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
            return $output >= $this->threshold ? 1.0 : $this->leakage;
        });
    }
}
