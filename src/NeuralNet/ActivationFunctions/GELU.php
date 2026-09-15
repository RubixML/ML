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
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix
    {
        return $x->map([$this, 'compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix
    {
        return $x->map([$this, '_differentiate']);
    }

    /**
     * @param float $x
     * @return float
     */
    public function compute(float $x) : float
    {
        return 0.5 * $x * (1.0 + tanh(self::ALPHA * ($x + self::BETA * $x ** 3)));
    }

    /**
     * @internal
     *
     * @param float $x
     * @return float
     */
    public function _differentiate(float $x) : float
    {
        $xHat = $x ** 3;

        $alpha = 0.0356774 * $xHat + self::ALPHA * $x;
        $beta = 0.0535161 * $xHat + 0.398942 * $x;

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
