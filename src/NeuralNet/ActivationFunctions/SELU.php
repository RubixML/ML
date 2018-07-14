<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

/**
 * SELU
 *
 * Scaled Exponential Linear Unit is a self-normalizing activation function
 * based on ELU.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SELU implements Rectifier
{
    /**
     * The scaling factor.
     *
     * @var float
     */
    protected $scale;

    /**
     * At which negative value the SELU will saturate. i.e. alpha = 1.0 means
     * that the leakage will never be more than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param  float  $scale
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $scale = 1.05070, float $alpha = 1.67326)
    {
        if ($scale < 1) {
            if ($alpha < 0) {
                throw new InvalidArgumentException('Scale parameter must be'
                    . ' greater than 1.');
            }
        }

        if ($alpha < 0) {
            throw new InvalidArgumentException('Alpha parameter must be a'
                . ' positive value.');
        }

        $this->scale = $scale;
        $this->alpha = $alpha;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [-($this->scale * $this->alpha), INF];
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
            return $value >= 0.0
                ? $this->scale * $value
                : $this->scale * $this->alpha * (exp($value) - 1);
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
        return $computed->map(function ($activation) {
            return $activation >= 0.0
                ? $this->scale * 1.0
                : $this->scale * $activation + 1;
        });
    }
}
