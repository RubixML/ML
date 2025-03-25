<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ELU;

use NumPower;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
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
class ELU implements ActivationFunction
{
    /**
     * @param float $alpha At which negative value the ELU will saturate. For example if alpha
     *              equals 1, the leaked value will never be greater than -1.0.
     * 
     * @throws InvalidArgumentException
     */
    public function __construct(protected float $alpha = 1.0)
    {
        if ($this->alpha < 0.0) {
            throw new InvalidAplhaException(
                message: "lpha must be greater than 0, $alpha given."
            );
        }
    }

    /**
     * @inheritdoc
     */
    public function activate(NumPower $input) : NumPower
    {
        return NumPower::maximum($input, 0) + (NumPower::exp(NumPower::minimum($input, 0)) - 1) * $this->alpha;
    }

    /**
     * @inheritdoc
     */
    public function differentiate(NumPower $input) : NumPower
    {
        return NumPower::greater($input, 0) + NumPower::lessEqual($input, 0) * NumPower::exp($input) * $this->alpha;
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
