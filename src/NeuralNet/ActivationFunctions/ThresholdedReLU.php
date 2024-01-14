<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

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
        return $input->greater($this->threshold);
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _activate(float $input) : float
    {
        return $input > $this->threshold ? $input : 0.0;
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
