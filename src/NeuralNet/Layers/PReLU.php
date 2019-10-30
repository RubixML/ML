<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameters\VectorParam;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use RuntimeException;
use Generator;

/**
 * PReLU
 *
 * Parametric Rectified Linear Units are leaky rectifiers whose leakage coefficient
 * is learned during training.
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
     * The parameterized leakage coeficients.
     *
     * @var \Rubix\ML\NeuralNet\Parameters\VectorParam|null
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
     * @throws \RuntimeException
     * @return int
     */
    public function width() : int
    {
        if (!$this->width) {
            throw new RuntimeException('Layer is not initialized.');
        }

        return $this->width;
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator
     */
    public function parameters() : Generator
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initlaized.');
        }

        yield $this->alpha;
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

        $alpha = $this->initializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->alpha = new VectorParam($alpha);

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
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
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return \Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initlaized.');
        }

        if (!$this->input) {
            throw new RuntimeException('Must perform a forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient->compute();

        $dIn = $this->input->clipUpper(0.);

        $dAlpha = $dOut->multiply($dIn)->sum();

        $this->alpha->update($optimizer->step($this->alpha, $dAlpha));

        $z = $this->input;

        unset($this->input);

        return new Deferred([$this, 'gradient'], [$z, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
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
     * Compute the leaky ReLU activation function and return a matrix.
     *
     * @param \Tensor\Matrix $z
     * @throws \RuntimeException
     * @return \Tensor\Matrix
     */
    protected function compute(Matrix $z) : Matrix
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer is not initialized.');
        }

        $alphas = $this->alpha->w();

        $computed = [];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            $activations = [];

            foreach ($row as $value) {
                $activations[] = $value > 0.
                    ? $value
                    : $alpha * $value;
            }

            $computed[] = $activations;
        }

        return Matrix::quick($computed);
    }

    /**
     * Calculate the partial derivatives of the activation function.
     *
     * @param \Tensor\Matrix $z
     * @throws \RuntimeException
     * @return \Tensor\Matrix
     */
    protected function differentiate(Matrix $z) : Matrix
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initlaized.');
        }

        $alphas = $this->alpha->w();

        $gradient = [];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            $temp = [];

            foreach ($row as $value) {
                $temp[] = $value > 0. ? 1. : $alpha;
            }

            $gradient[] = $temp;
        }

        return Matrix::quick($gradient);
    }

    /**
     * Read the parameters and return them in an associative array.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function read() : array
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initlaized.');
        }
        
        return [
            'alpha' => clone $this->alpha,
        ];
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param array $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->alpha = $parameters['alpha'];
    }
}
