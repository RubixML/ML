<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;

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
     */
    protected static function sech2(NDArray $value) : NDArray
    {
        $cosh = NumPower::cosh($value);

        $sech = 1.0 / $cosh;

        return $sech ** 2;
    }

    /**
     * @inheritdoc
     */
    public function activate(NDArray $input) : NDArray
    {
        return 0.5 * $input * (1.0 + NumPower::tanh(self::ALPHA * ($input + self::BETA * $input ** 3)));
    }

    /**
     * @inheritdoc
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray
    {
        //input
        $zHat = $input ** 3;

        $alpha = 0.0356774 * $zHat + self::ALPHA * $input;
        $beta = 0.0535161 * $zHat + 0.398942 * $input;

        return 0.5 * NumPower::tanh($alpha) + $beta * self::sech2($alpha) + 0.5;
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
     * Return the string representation of the activation function.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'GELU';
    }
}
