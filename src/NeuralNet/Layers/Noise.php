<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use Closure;

/**
 * Noise
 *
 * This layer adds random Gaussian noise to the inputs to the layer with a
 * standard deviation given as a parameter. Noise added to neural network
 * activations acts as a regularizer by indirectly adding a penalty to the
 * weights through the cost function in the output layer.
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
     * The standard devaiation of the gaussian noise to add to the inputs.
     *
     * @var float
     */
    protected $stddev;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * @param float $stddev
     * @throws \InvalidArgumentException
     */
    public function __construct(float $stddev = 0.1)
    {
        if ($stddev < 0.) {
            throw new InvalidArgumentException('Standard deviation must be'
                . " 0 or greater, $stddev given.");
        }

        $this->stddev = $stddev;
    }

    /**
     * @return int|null
     */
    public function width() : ?int
    {
        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
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
     * @param \Rubix\Tensor\Matrix $input
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $noise = Matrix::gaussian(...$input->shape())
            ->multiply($this->stddev);

        return $input->add($noise);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $input;
    }

    /**
     * Calculate the gradients of the layer and update the parameters.
     *
     * @param Closure $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return Closure
     */
    public function back(Closure $prevGradient, Optimizer $optimizer) : Closure
    {
        return $prevGradient;
    }
}
