<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Activation
 *
 * Activation layers apply a user-defined non-linear activation function to their
 * inputs.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Activation implements Hidden
{
    /**
     * The function that computes the output of the layer.
     *
     * @var ActivationFunction
     */
    protected ActivationFunction $activationFn;

    /**
     * The width of the layer.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The memorized input matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $x = null;

    /**
     * The memorized activation matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $z = null;

    /**
     * @param ActivationFunction $activationFn
     */
    public function __construct(ActivationFunction $activationFn)
    {
        $this->activationFn = $activationFn;
    }

    /**
     * Return the width of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return positive-int
     */
    public function width() : int
    {
        if ($this->width === null) {
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
     * @param positive-int $fanIn
     * @return positive-int
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
     * @param Matrix $x
     * @return Matrix
     */
    public function forward(Matrix $x) : Matrix
    {
        $z = $this->activationFn->activate($x);

        $this->x = $x;
        $this->z = $z;

        return $z;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @internal
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function infer(Matrix $x) : Matrix
    {
        return $this->activationFn->activate($x);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @throws RuntimeException
     * @return Deferred
     */
    public function back(Deferred $prevGradient) : Deferred
    {
        if (!$this->x or !$this->z) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $x = $this->x;
        $z = $this->z;

        $this->x = $this->z = null;

        return new Deferred([$this, 'gradient'], [$x, $z, $prevGradient]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @param Deferred $prevGradient
     * @return Matrix
     */
    public function gradient(Matrix $x, Matrix $z, Deferred $prevGradient) : Matrix
    {
        return $this->activationFn->differentiate($x, $z)->multiply($prevGradient());
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
        return "Activation (activation fn: {$this->activationFn})";
    }
}
