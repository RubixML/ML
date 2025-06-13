<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ELU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\SingleBufferDerivative;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\Exceptions\InvalidAplhaException;

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
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */
class ELU implements ActivationFunction, SingleBufferDerivative
{
    /**
     * @param float $alpha At which negative value the ELU will saturate. For example if alpha
     *              equals 1, the leaked value will never be greater than -1.0.
     * 
     * @throws InvalidAplhaException
     */
    public function __construct(protected float $alpha = 1.0)
    {
        if ($this->alpha < 0.0) {
            throw new InvalidAplhaException(
                message: "Alpha must be greater than 0, $alpha given."
            );
        }
    }

    public function activate(NDArray $input) : NDArray
    {
        $a = NumPower::multiply(
            a: NumPower::expm1(NumPower::minimum(
                a: $input,
                b: 0
            )),
            b: $this->alpha
        );

        return NumPower::add(
            a: NumPower::maximum(
                a: $input,
                b: 0
            ),
            b: $a
        );
    }

    /**
     * Calculate the derivative of the activation output
     *
     * @param NDArray $x Output matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $x) : NDArray
    {
        $a = NumPower::multiply(
            a: NumPower::lessEqual(a: $x, b: 0),
            b: NumPower::exp($x)
        );
        $b = NumPower::multiply(
            a: $a,
            b: $this->alpha
        );

        return NumPower::add(
            a: NumPower::greater(
                a: $x,
                b: 0
            ),
            b: $b
        );
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return "ELU (alpha: {$this->alpha})";
    }
}
