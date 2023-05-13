<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

use function tanh;
use function cosh;

/**
 * GELU
 *
 * Gaussian Error Linear Units (GELUs) are rectifiers that are gated by the magnitude of their input rather
 * than the sign of their input as with ReLU variants. Their output can be interpreted as the expected value
 * of a neuron with random dropout regularization applied.
 *
 * [1] D. Hendrycks et al. (2018). Gaussian Error Linear Units (GELUs).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GELU implements ActivationFunction
{
    /**
     * The square root of two over pi.
     *
     * @var float
     */
    protected const ALPHA = 0.7978845608;

    /**
     * Gaussian error function approximation term.
     *
     * @var float
     */
    protected const BETA = 0.044715;

    /**
     * Three WTF constants.
     */
    protected const TAU = [
        0.0356774, 0.0535161, 0.398942,
    ];

    /**
     * Calculate the squared hyperbolic secant of a number.
     *
     * @param float $value
     * @return float
     */
    protected static function sech2(float $value) : float
    {
        $cosh = cosh($value);

        if ($cosh === 0.0) {
            return 0.0;
        }

        $sech = 1.0 / $cosh;

        return $sech ** 2;
    }

    /**
     * Compute the output value.
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function activate(Matrix $z) : Matrix
    {
        return $z->map([$this, 'compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $z->map([$this, '_differentiate']);
    }

    /**
     * @param float $z
     * @return float
     */
    public function compute(float $z) : float
    {
        return 0.5 * $z * (1.0 + tanh(self::ALPHA * ($z + self::BETA * $z ** 3)));
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _differentiate(float $z) : float
    {
        $alpha = self::TAU[0] * $z ** 3 + self::ALPHA * $z;

        $beta = self::TAU[1] * $z ** 3 + self::TAU[2] * $z;

        return 0.5 * tanh($alpha) + $beta * self::sech2($alpha) + 0.5;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'GELU';
    }
}
