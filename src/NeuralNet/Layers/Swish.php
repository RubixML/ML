<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
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
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected \Rubix\ML\NeuralNet\Initializers\Initializer $initializer;

    /**
     * The sigmoid activation function.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid
     */
    protected \Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid $sigmoid;

    /**
     * The width of the layer.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The parameterized scaling factors.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected ?\Rubix\ML\NeuralNet\Parameter $beta = null;

    /**
     * The memoized input matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected ?\Tensor\Matrix $input = null;

    /**
     * The memorized activation matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected ?\Tensor\Matrix $output = null;

    /**
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $initializer
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
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

        $beta = $this->initializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->width = $fanOut;
        $this->beta = new Parameter($beta);

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
        $output = $this->activate($input);

        $this->input = $input;
        $this->output = $output;

        return $output;
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
        return $this->activate($input);
    }

    /**
     * Calculate the gradient and update the parameters of the layer.
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
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input or !$this->output) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $dOut = $prevGradient();

        $dIn = $this->input;

        $dBeta = $dOut->multiply($dIn)->sum();

        $this->beta->update($dBeta, $optimizer);

        $input = $this->input;
        $output = $this->output;

        $this->input = $this->output = null;

        return new Deferred([$this, 'gradient'], [$input, $output, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $dOut
     * @return \Tensor\Matrix
     */
    public function gradient($input, $output, $dOut) : Matrix
    {
        return $this->differentiate($input, $output)
            ->multiply($dOut);
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws \RuntimeException
     * @return \Generator<\Rubix\ML\NeuralNet\Parameter>
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
     * @param \Rubix\ML\NeuralNet\Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
    }

    /**
     * Compute the Swish activation function and return a matrix.
     *
     * @param \Tensor\Matrix $input
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
     */
    protected function activate(Matrix $input) : Matrix
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $zHat = $input->multiply($this->beta->param());

        return $this->sigmoid->activate($zHat)
            ->multiply($input);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $output
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
     */
    protected function differentiate(Matrix $input, Matrix $output) : Matrix
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $ones = Matrix::ones(...$output->shape());

        return $output->divide($input)
            ->multiply($ones->subtract($output))
            ->add($output);
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
