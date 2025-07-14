<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU\Exceptions\InvalidThresholdException;

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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class ThresholdedReLU implements ActivationFunction, IBufferDerivative
{
    /**
     * The input value necessary to trigger an activation.
     *
     * @var float
     */
    protected float $threshold;

    /**
     * Class constructor.
     *
     * @param float $threshold The input value necessary to trigger an activation.
     * @throws InvalidThresholdException
     */
    public function __construct(float $threshold = 1.0)
    {
        if ($threshold < 0.0) {
            throw new InvalidThresholdException(
                message: "Threshold must be positive, $threshold given."
            );
        }

        $this->threshold = $threshold;
    }

    /**
     * Compute the activation.
     *
     * f(x) = x if x > threshold, 0 otherwise
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        // Create a mask where input > threshold
        $mask = NumPower::greater($input, $this->threshold);

        // Apply the mask to the input
        return NumPower::multiply($input, $mask);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * f'(x) = 1 if x > threshold, 0 otherwise
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function differentiate(NDArray $input) : NDArray
    {
        // The derivative is 1 where input > threshold, 0 otherwise
        return NumPower::greater($input, $this->threshold);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Thresholded ReLU (threshold: {$this->threshold})";
    }
}
