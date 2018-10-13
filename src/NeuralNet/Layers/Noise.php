<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;

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
class Noise implements Hidden, Nonparametric
{
    /**
     * The amount of gaussian noise to add to the inputs i.e the standard
     * deviation of the noise.
     *
     * @var float
     */
    protected $amount;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * @param  float  $amount
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $amount = 0.1)
    {
        if ($amount < 0.) {
            throw new InvalidArgumentException('Noise parameter must be'
                . '0 or greater.');
        }

        $this->amount = $amount;
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
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Generate a random noise matrix and add it to the input.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $noise = Matrix::gaussian(...$input->shape())
            ->multiply($this->amount);

        return $input->add($noise);
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $input;
    }

    /**
     * Calculate the gradients of the layer and update the parameters.
     *
     * @param  callable  $prevGradient
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return callable
     */
    public function back(callable $prevGradient, Optimizer $optimizer) : callable
    {
        return $prevGradient;
    }
}
