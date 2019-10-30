<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use InvalidArgumentException;

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
    protected $threshold;

    /**
     * @param float $threshold
     * @throws \InvalidArgumentException
     */
    public function __construct(float $threshold = 1.)
    {
        if ($threshold < 0.) {
            throw new InvalidArgumentException('Threshold must be'
                . " positive, $threshold given.");
        }

        $this->threshold = $threshold;
    }

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0, INF];
    }

    /**
     * Compute the output value.
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
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $z->greater($this->threshold);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > $this->threshold ? $z : 0.;
    }
}
