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
     * The amount of leakage as a ratio of the input value to allow to pass through when inactivated.
     *
     * @var float
     */
    protected float $leakage;

    /**
     * @param float $leakage
     * @throws InvalidArgumentException
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
     * Compute the activation.
     *
     * @internal
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function activate(Matrix $input) : Matrix
    {
        return $input->map([$this, '_activate']);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param Matrix $input
     * @param Matrix $output
     * @return Matrix
     */
    public function differentiate(Matrix $input, Matrix $output) : Matrix
    {
        return $input->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _activate(float $input) : float
    {
        return $input > 0.0
            ? $input
            : $this->leakage * $input;
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _differentiate(float $input) : float
    {
        return $input > 0.0
            ? 1.0
            : $this->leakage;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Leaky ReLU (leakage: {$this->leakage})";
    }
}
