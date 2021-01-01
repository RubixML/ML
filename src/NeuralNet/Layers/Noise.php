<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Noise
 *
 * This layer adds random Gaussian noise to the inputs to the layer with a
 * given standard deviation. Noise added to neural network activations acts as
 * a regularizer by indirectly adding a penalty to the weights through the cost
 * function in the output layer.
 *
 * References:
 * [1] C. Gulcehre et al. (2016). Noisy Activation Functions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Noise implements Hidden
{
    /**
     * The amount (standard deviation) of the gaussian noise to add to the inputs.
     *
     * @var float
     */
    protected $stdDev;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * @param float $stdDev
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $stdDev)
    {
        if ($stdDev < 0.0) {
            throw new InvalidArgumentException('Standard deviation must'
                . " be 0 or greater, $stdDev given.");
        }

        $this->stdDev = $stdDev;
    }

    /**
     * Return the width of the layer.
     *
     * @internal
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return int
     */
    public function width() : int
    {
        if (!$this->width) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @internal
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $noise = Matrix::gaussian(...$input->shape())
            ->multiply($this->stdDev);

        return $input->add($noise);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $input;
    }

    /**
     * Calculate the gradients of the layer and update the parameters.
     *
     * @internal
     *
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return \Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        return $prevGradient;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Noise (std_dev: {$this->stdDev})";
    }
}
