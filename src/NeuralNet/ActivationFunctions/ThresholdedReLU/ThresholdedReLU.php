<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Thresholded ReLU
 *
 * A Thresholded ReLU (Rectified Linear Unit) only outputs the signal above
 * a user-defined threshold parameter.
 *
 * References:
 * [1] K. Konda et al. (2015). Zero-bias Autoencoders and the Benefits of
 * Co-adapting Features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ThresholdedReLU implements ActivationFunction
{
    /**
     * The input value necessary to trigger an activation.
     *
     * @var float
     */
    protected float $threshold;

    /**
     * @param float $threshold
     * @throws InvalidArgumentException
     */
    public function __construct(float $threshold = 1.0)
    {
        if ($threshold < 0.0) {
            throw new InvalidArgumentException('Threshold must be'
                . " positive, $threshold given.");
        }

        $this->threshold = $threshold;
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
        return NumPower::greater($input, $this->threshold) * $input;
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
        return NumPower::greater($input, $this->threshold);
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
        return "Thresholded ReLU (threshold: {$this->threshold})";
    }
}
