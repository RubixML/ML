<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Parameter;
use Generator;

use const Rubix\ML\EPSILON;

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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
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
     * @var NDArray|null
     */
    protected ?NDArray $input = null;

    /**
     * The memorized activation matrix.
     *
     * @var NDArray|null
     */
    protected ?NDArray $output = null;

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

        // Initialize beta as a vector of length fanOut (one beta per neuron)
        // Using shape [fanOut, 1] then flattening to [fanOut]
        $betaMat = $this->initializer->initialize(1, $fanOut);
        $beta = NumPower::flatten($betaMat);

        $this->width = $fanOut;
        $this->beta = new Parameter($beta);

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

        $this->output = $this->activate($input);

        return $this->output;
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
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input or !$this->output) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        /** @var NDArray $dOut */
        $dOut = $prevGradient();

        // Gradient of the loss with respect to beta
        // dL/dbeta = sum_over_batch(dL/dy * dy/dbeta)
        // Here we use a simplified formulation: dL/dbeta ~ sum(dOut * input)
        $dBetaFull = NumPower::multiply($dOut, $this->input);

        // Sum over the batch axis (axis = 1) to obtain a gradient vector [width]
        $dBeta = NumPower::sum($dBetaFull, axis: 1);

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
     * @param NDArray $input
     * @param NDArray $output
     * @param NDArray $dOut
     * @return NDArray
     */
    public function gradient(NDArray $input, NDArray $output, NDArray $dOut) : NDArray
    {
        $derivative = $this->differentiate($input, $output);

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
     * @param NDArray $input
     * @throws RuntimeException
     * @return NDArray
     */
    protected function activate(NDArray $input) : NDArray
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        // Reshape beta vector [width] to column [width, 1] for broadcasting
        $betaCol = NumPower::reshape($this->beta->param(), [$this->width(), 1]);

        $zHat = NumPower::multiply($betaCol, $input);

        $activated = $this->sigmoid->activate($zHat);

        return NumPower::multiply($activated, $input);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     * Formulation: derivative = (output / input) * (1 - output) + output
     *
     * @param NDArray $input
     * @param NDArray $output
     * @throws RuntimeException
     * @return NDArray
     */
    protected function differentiate(NDArray $input, NDArray $output) : NDArray
    {
        if (!$this->beta) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        // Prevent division by zero if the input contains zero values
        $denominator = NumPower::add($input, EPSILON);
        $term1 = NumPower::divide($output, $denominator);

        $oneMinusOutput = NumPower::subtract(1.0, $output);
        $product = NumPower::multiply($term1, $oneMinusOutput);

        return NumPower::add($product, $output);
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
