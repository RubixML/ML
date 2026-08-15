<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\Initializers\HeUniform;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Deferred;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

/**
 * Dense
 *
 * Dense (or *fully connected*) hidden layers are layers of neurons that connect to each node
 * in the previous layer by a parameterized synapse. They perform a linear transformation on
 * their input and are usually followed by an Activation layer. The majority of the trainable
 * parameters in a standard feed-forward neural network are contained within Dense hidden layers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Dense implements Hidden, Parametric
{
    /**
     * The number of nodes in the layer.
     *
     * @var positive-int
     */
    protected int $neurons;

    /**
     * The amount of L2 regularization applied to the weights.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * Should the layer include a bias parameter?
     *
     * @var bool
     */
    protected bool $bias;

    /**
     * The weight initializer.
     *
     * @var Initializer
     */
    protected Initializer $weightInitializer;

    /**
     * The bias initializer.
     *
     * @var Initializer
     */
    protected Initializer $biasInitializer;

    /**
     * The weights.
     *
     * @var Parameter|null
     */
    protected ?Parameter $weights = null;

    /**
     * The biases.
     *
     * @var Parameter|null
     */
    protected ?Parameter $biases = null;

    /**
     * The memorized inputs to the layer.
     *
     * @var NDArray|null
     */
    protected ?NDArray $input = null;

    /**
     * @param int $neurons
     * @param float $l2Penalty
     * @param bool $bias
     * @param Initializer|null $weightInitializer
     * @param Initializer|null $biasInitializer
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $neurons,
        float $l2Penalty = 0.0,
        bool $bias = true,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        if ($neurons < 1) {
            throw new InvalidArgumentException("Number of neurons must be greater than 0, $neurons given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException("L2 Penalty must be greater than 0, $l2Penalty given.");
        }

        $this->neurons = $neurons;
        $this->l2Penalty = $l2Penalty;
        $this->bias = $bias;
        $this->weightInitializer = $weightInitializer ?? new HeUniform();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.0);

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
    }

    /**
     * Return the width of the layer.
     *
     * @internal
     *
     * @return positive-int
     */
    public function width() : int
    {
        return $this->neurons;
    }

    /**
     * Return the weight matrix.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return NDArray
     */
    public function weights() : NDArray
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        return $this->weights->param();
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
        $fanOut = $this->neurons;

        $weights = $this->weightInitializer->initialize($fanIn, $fanOut);

        $this->weights = new Parameter($weights);

        if ($this->bias) {
            // Initialize biases as a vector of length fanOut
            $biasMat = $this->biasInitializer->initialize(1, $fanOut);
            $biases = NumPower::flatten($biasMat);

            $this->biases = new Parameter($biases);
        }

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     * @internal
     */
    public function forward(NDArray $input) : NDArray
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $output = NumPower::matmul($this->weights->param(), $input);

        if ($this->biases) {
            // Reshape bias vector [fanOut] to column [fanOut, 1] to match output [fanOut, n]
            $bias = NumPower::reshape($this->biases->param(), [$this->neurons, 1]);
            // Manual “broadcast”: [neurons, n] + [neurons, 1]
            $output = NumPower::add($output, $bias);
        }

        $this->input = $input;

        return $output;
    }

    /**
     * Compute an inference pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     * @internal
     */
    public function infer(NDArray $input) : NDArray
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $output = NumPower::matmul($this->weights->param(), $input);

        if ($this->biases) {
            // Reshape bias vector [fanOut] to column [fanOut, 1] to match output [fanOut, n]
            $bias = NumPower::reshape($this->biases->param(), [$this->neurons, 1]);
            // Manual “broadcast”: [neurons, n] + [neurons, 1]
            $output = NumPower::add($output, $bias);
        }

        return $output;
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
        if (!$this->weights) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        /** @var NDArray $dOut */
        $dOut = $prevGradient();

        $inputT = NumPower::transpose($this->input, [1, 0]);

        $dW = NumPower::matmul($dOut, $inputT);

        $weights = $this->weights->param();

        if ($this->l2Penalty) {
            $dW = NumPower::add(
                $dW,
                NumPower::multiply($weights, $this->l2Penalty)
            );
        }

        $this->weights->update($dW, $optimizer);

        if ($this->biases) {
            // Sum gradients over the batch dimension to obtain a bias gradient
            // with the same shape as the bias vector [neurons]
            $dB = NumPower::sum($dOut, axis: 1);

            $this->biases->update($dB, $optimizer);
        }

        $this->input = null;

        return new Deferred([$this, 'gradient'], [$weights, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param NDArray $weights
     * @param NDArray $dOut
     * @return NDArray
     */
    public function gradient(NDArray $weights, NDArray $dOut) : NDArray
    {
        $weightsT = NumPower::transpose($weights, [1, 0]);

        return NumPower::matmul($weightsT, $dOut);
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'weights' => $this->weights;

        if ($this->biases) {
            yield 'biases' => $this->biases;
        }
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
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'] ?? null;
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
        return "Dense (neurons: {$this->neurons}, l2 penalty: {$this->l2Penalty},"
            . ' bias: ' . Params::toString($this->bias) . ','
            . " weight initializer: {$this->weightInitializer},"
            . " bias initializer: {$this->biasInitializer})";
    }
}
