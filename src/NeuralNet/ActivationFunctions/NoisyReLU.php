<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use MathPHP\LinearAlgebra\Matrix;
use InvalidArgumentException;

/**
 * Noisy ReLU
 *
 * Noisy ReLU neurons emit Gaussian noise with a standard deviation given by the
 * noise parameter along with their activation. Noise in a neural network acts
 * as a regularizer by adding a penalty to the weights through the cost function
 * in the output layer.
 *
 * References:
 * [1] C. Gulcehre et al. (2016). Noisy Activation Functions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NoisyReLU implements Rectifier
{
    const PHI = 100000000;

    /**
     * The scaled minimum gaussian noise value.
     *
     * @var int
     */
    protected $min;

    /**
     * The scaled maximum gaussian noise value.
     *
     * @var int
     */
    protected $max;

    /**
     * @param  float  $noise
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $noise = 0.1)
    {
        if ($noise < 0.) {
            throw new InvalidArgumentException('Noise parameter must be'
                . '0 or greater.');
        }

        $this->max = (int) round($noise * self::PHI);
        $this->min = -$this->max;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return array
     */
    public function range() : array
    {
        return [$this->min / self::PHI, INF];
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
            $noise = rand($this->min, $this->max) / self::PHI;

            return $value > 0. ? max(0., $value + $noise) : -abs($noise);
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
            return $activation > 0. ? 1. : 0.;
        });
    }
}
