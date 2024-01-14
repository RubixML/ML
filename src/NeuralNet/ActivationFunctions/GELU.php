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
     * @param Matrix $z
     * @return Matrix
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
     * @param Matrix $z
     * @param Matrix $computed
     * @return Matrix
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
        $zHat = $z ** 3;

        $alpha = 0.0356774 * $zHat + self::ALPHA * $z;
        $beta = 0.0535161 * $zHat + 0.398942 * $z;

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
