<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Leaky ReLU
 *
 * Leaky Rectified Linear Units are functions that output x when x > 0 or a
 * small leakage value when x < 0. The amount of leakage is controlled by the
 * user-specified parameter.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network
 * Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeakyReLU implements ActivationFunction
{
    /**
     * The amount of leakage as a ratio of the input value to allow to pass
     * through when not activated.
     *
     * @var float
     */
    protected $leakage;

    /**
     * @param float $leakage
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $leakage = 0.1)
    {
        if ($leakage <= 0.0 or $leakage >= 1.0) {
            throw new InvalidArgumentException('Leakage must be between'
                . " 0 and 1, $leakage given.");
        }

        $this->leakage = $leakage;
    }

    /**
     * Compute the output value.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
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
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0.0 ? $z : $this->leakage * $z;
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _differentiate(float $z) : float
    {
        return $z > 0.0 ? 1.0 : $this->leakage;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Leaky ReLU (leakage: {$this->leakage})";
    }
}
