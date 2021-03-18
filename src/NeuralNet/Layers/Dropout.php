<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Dropout
 *
 * Dropout is a regularization technique for reducing overfitting in neural
 * networks by preventing complex co-adaptations on training data. It works
 * by temporarily disabling neurons during each training pass. It also is a
 * very efficient way of performing model averaging with neural networks.
 *
 * References:
 * [1] N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural
 * Networks from Overfitting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dropout implements Hidden
{
    /**
     * The ratio of neurons that are dropped during each training pass.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    protected $scale;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The memoized dropout mask.
     *
     * @var \Tensor\Matrix|null
     */
    protected $mask;

    /**
     * @param float $ratio
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $ratio = 0.5)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $this->ratio = $ratio;
        $this->scale = 1.0 / (1.0 - $ratio);
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
        $this->mask = Matrix::rand(...$input->shape())
            ->greater($this->ratio)
            ->multiply($this->scale);

        return $this->mask->multiply($input);
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->mask) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $mask = $this->mask;

        $this->mask = null;

        return new Deferred([$this, 'gradient'], [$prevGradient, $mask]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Tensor\Matrix $mask
     * @return \Tensor\Matrix
     */
    public function gradient(Deferred $prevGradient, Matrix $mask)
    {
        return $prevGradient()->multiply($mask);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Dropout (ratio: {$this->ratio})";
    }
}
