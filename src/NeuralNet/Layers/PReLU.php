<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

/**
 * PReLU
 *
 * Parametric Rectified Linear Units are leaky rectifiers whose leakage coefficients
 * are learned during training.
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PReLU implements Hidden, Parametric
{
    /**
     * The initializer of the alpha (leakage) parameter.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $initializer;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The parameterized leakage coefficients.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $alpha;

    /**
     * The memoized input matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $input;

    /**
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $initializer
     */
    public function __construct(?Initializer $initializer = null)
    {
        $this->initializer = $initializer ?? new Constant(0.25);
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

        $alpha = $this->initializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->alpha = new Parameter($alpha);

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
        $this->input = $input;

        return $this->compute($input);
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
        return $this->compute($input);
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
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $dOut = $prevGradient();

        $dIn = $this->input->clipUpper(0.0);

        $dAlpha = $dOut->multiply($dIn)->sum();

        $this->alpha->update($optimizer->step($this->alpha, $dAlpha));

        $z = $this->input;

        $this->input = null;

        return new Deferred([$this, 'gradient'], [$z, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $dOut
     * @return \Tensor\Matrix
     */
    public function gradient($z, $dOut) : Matrix
    {
        return $this->differentiate($z)->multiply($dOut);
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
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'alpha' => $this->alpha;
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
        $this->alpha = $parameters['alpha'];
    }

    /**
     * Compute the leaky ReLU activation function and return a matrix.
     *
     * @param \Tensor\Matrix $z
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
     */
    protected function compute(Matrix $z) : Matrix
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $alphas = $this->alpha->param()->asArray();

        $computed = [];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            $activations = [];

            foreach ($row as $value) {
                $activations[] = $value > 0.0
                    ? $value
                    : $alpha * $value;
            }

            $computed[] = $activations;
        }

        return Matrix::quick($computed);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \Tensor\Matrix $z
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
     */
    protected function differentiate(Matrix $z) : Matrix
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $alphas = $this->alpha->param()->asArray();

        $gradient = [];

        foreach ($z as $i => $row) {
            $leakage = $alphas[$i];

            $derivative = [];

            foreach ($row as $value) {
                $derivative[] = $value > 0.0 ? 1.0 : $leakage;
            }

            $gradient[] = $derivative;
        }

        return Matrix::quick($gradient);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "PReLU (alpha_initializer: {$this->initializer})";
    }
}
