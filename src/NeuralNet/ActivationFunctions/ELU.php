<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

/**
 * ELU
 *
 * Exponential Linear Units are a type of rectifier that soften the transition
 * from non-activated to activated using the exponential function.
 *
 * References:
 * [1] D. A. Clevert et al. (2016). Fast and Accurate Deep Network Learning by
 * Exponential Linear Units.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ELU implements ActivationFunction
{
    /**
     * At which negative value the ELU will saturate. i.e. alpha = 1.means
     * that the leakage will never be more than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1.)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('Alpha cannot be less than'
                . " 0, $alpha given.");
        }

        $this->alpha = $alpha;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-$this->alpha, INF];
    }

    /**
     * Compute the output value.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @param  \Rubix\Tensor\Matrix  $computed
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param  float  $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0. ? $z : $this->alpha * (exp($z) - 1.);
    }

    /**
     * @param  float  $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed > 0. ? 1. : $computed + $this->alpha;
    }
}
