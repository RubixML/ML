<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

/**
 * Swish
 *
 * Swish is a parametric activation layer that utilizes smooth rectified activation functions. The trainable
 * *beta* parameter allows each activation function in the layer to tailor its output to the training set by
 * interpolating between the linear function and ReLU.
 *
 * [1] P. Ramachandran et al. (2017). Swish: A Self-gated Activation Function.
 * [2] P. Ramachandran et al. (2017). Searching for Activation Functions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Swish implements Hidden, Parametric
{
    /**
     * The initializer of the beta parameter.
     *
     * @var Initializer
     */
    protected Initializer $initializer;

    /**
     * The sigmoid activation function.
     *
     * @var Sigmoid
     */
    protected Sigmoid $sigmoid;

    /**
     * The width of the layer.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The parameterized scaling factors.
     *
     * @var Parameter|null
     */
    protected ?Parameter $beta = null;

    /**
     * The memoized input matrix.
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
     * @param Initializer|null $initializer
     */
    public function __construct(?Initializer $initializer = null)
    {
        $this->initializer = $initializer ?? new Constant(1.0);
        $this->sigmoid = new Sigmoid();
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
        $this->beta = $this->initializer->initialize([$fanOut]);

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
        $z = $this->activate($x);

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
        return $this->activate($x);
    }

    /**
     * Calculate the gradient for the previous layer and record the gradient of
     * the parameters of this layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @throws RuntimeException
     * @return Deferred
     */
    public function back(Deferred $prevGradient) : Deferred
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->x or !$this->z) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $dOut = $prevGradient->compute();

        $x = $this->x;
        $z = $this->z;

        $dX = $x->multiply($z)->subtract($z->square());

        $dBeta = $dOut->multiply($dX)->sum();

        $beta = $this->beta->param();

        $this->beta->accumulateGradient($dBeta);

        $this->x = $this->z = null;

        return new Deferred([$this, 'gradient'], [$x, $z, $dOut, $beta]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @param Matrix $dOut
     * @param Vector $beta
     * @return Matrix
     */
    public function gradient($x, $z, $dOut, $beta) : Matrix
    {
        return $this->differentiate($x, $z, $beta)->multiply($dOut);
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws \RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'beta' => $this->beta;
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @internal
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
    }

    /**
     * Compute the Swish activation function and return a matrix.
     *
     * @param Matrix $x
     * @throws RuntimeException
     * @return Matrix
     */
    protected function activate(Matrix $x) : Matrix
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $zHat = $x->multiply($this->beta->param());

        return $this->sigmoid->activate($zHat)->multiply($x);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     *     f'(x) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
     *
     * where z = beta * x. This formulation is defined at x = 0 and for any
     * value of beta.
     *
     * @param Matrix $x
     * @param Matrix $z
     * @param Vector $beta
     * @return Matrix
     */
    protected function differentiate(Matrix $x, Matrix $z, Vector $beta) : Matrix
    {
        $zHat = $x->multiply($beta);

        $sigmoid = $this->sigmoid->activate($zHat);

        $ones = Matrix::ones(...$z->shape());

        $term = $zHat->multiply($sigmoid)->multiply($ones->subtract($sigmoid));

        return $sigmoid->add($term);
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
        return "Swish (initializer: {$this->initializer})";
    }
}
