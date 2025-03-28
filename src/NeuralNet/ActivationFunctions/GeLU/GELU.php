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
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
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
        $sech2 = (1 / NumPower::cosh($value)) ** 2;

        return 0.5 * NumPower::tanh($alpha) + $beta * $sech2 + 0.5;
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
