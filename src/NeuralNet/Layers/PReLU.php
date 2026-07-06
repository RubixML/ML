<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Parameter;
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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class PReLU implements Hidden, Parametric
{
    /**
     * The initializer of the alpha (leakage) parameter.
     *
     * @var Initializer
     */
    protected Initializer $initializer;

    /**
     * The width of the layer.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The parameterized leakage coefficients.
     *
     * @var Parameter|null
     */
    protected ?Parameter $alpha = null;

    /**
     * The memoized input matrix.
     *
     * @var NDArray|null
     */
    protected ?NDArray $input = null;

    /**
     * @param Initializer|null $initializer
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

        // Initialize alpha as a vector of length fanOut (one alpha per neuron)
        // Using shape [fanOut, 1] then flattening to [fanOut]
        $alphaMat = $this->initializer->initialize(1, $fanOut);
        $alpha = NumPower::flatten($alphaMat);

        $this->width = $fanOut;
        $this->alpha = new Parameter($alpha);

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray
    {
        $this->input = $input;

        return $this->activate($input);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        return $this->activate($input);
    }

    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        /** @var NDArray $dOut */
        $dOut = $prevGradient();

        // Negative part of the input (values <= 0), used for dL/dalpha
        $negativeInput = NumPower::minimum($this->input, 0.0);

        $dAlphaFull = NumPower::multiply($dOut, $negativeInput);

        // Sum over the batch axis (axis = 1) to obtain a gradient vector [width]
        $dAlpha = NumPower::sum($dAlphaFull, axis: 1);

        $this->alpha->update($dAlpha, $optimizer);

        $input = $this->input;

        $this->input = null;

        return new Deferred([$this, 'gradient'], [$input, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param NDArray $input
     * @param NDArray $dOut
     * @return NDArray
     */
    public function gradient(NDArray $input, NDArray $dOut) : NDArray
    {
        $derivative = $this->differentiate($input);

        return NumPower::multiply($derivative, $dOut);
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
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->alpha = $parameters['alpha'];
    }

    /**
     * Compute the leaky ReLU activation function and return a matrix.
     *
     * @param NDArray $input
     * @throws RuntimeException
     * @return NDArray
     */
    protected function activate(NDArray $input) : NDArray
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        // Reshape alpha vector [width] to column [width, 1] for broadcasting
        $alphaCol = NumPower::reshape($this->alpha->param(), [$this->width(), 1]);

        $positiveActivation = NumPower::maximum($input, 0.0);

        $negativeActivation = NumPower::multiply(
            NumPower::minimum($input, 0.0),
            $alphaCol,
        );

        return NumPower::add($positiveActivation, $negativeActivation);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param NDArray $input
     * @throws RuntimeException
     * @return NDArray
     */
    protected function differentiate(NDArray $input) : NDArray
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        // Reshape alpha vector [width] to column [width, 1] for broadcasting
        $alphaCol = NumPower::reshape($this->alpha->param(), [$this->width(), 1]);

        $positivePart = NumPower::greater($input, 0.0);

        $negativePart = NumPower::multiply(
            NumPower::lessEqual($input, 0.0),
            $alphaCol,
        );

        return NumPower::add($positivePart, $negativePart);
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
        return "PReLU (initializer: {$this->initializer})";
    }
}
