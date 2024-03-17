<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\He;
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
     * @var Matrix|null
     */
    protected ?Matrix $input = null;

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
            throw new InvalidArgumentException('Number of neurons'
                . " must be greater than 0, $neurons given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException('L2 Penalty must be'
                . " greater than 0, $l2Penalty given.");
        }

        $this->neurons = $neurons;
        $this->l2Penalty = $l2Penalty;
        $this->bias = $bias;
        $this->weightInitializer = $weightInitializer ?? new He();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.0);
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
     * @return Matrix
     */
    public function weights() : Matrix
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
            $biases = $this->biasInitializer->initialize(1, $fanOut)->columnAsVector(0);

            $this->biases = new Parameter($biases);
        }

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @internal
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $output = $this->weights->param()->matmul($input);

        if ($this->biases) {
            $output = $output->add($this->biases->param());
        }

        $this->input = $input;

        return $output;
    }

    /**
     * Compute an inference pass through the layer.
     *
     * @internal
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $output = $this->weights->param()->matmul($input);

        if ($this->biases) {
            $output = $output->add($this->biases->param());
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
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $dOut = $prevGradient();

        $dW = $dOut->matmul($this->input->transpose());

        $weights = $this->weights->param();

        if ($this->l2Penalty) {
            $dW = $dW->add($weights->multiply($this->l2Penalty));
        }

        $this->weights->update($dW, $optimizer);

        if ($this->biases) {
            $dB = $dOut->sum();

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
     * @param Matrix $weights
     * @param Matrix $dOut
     * @return Matrix
     */
    public function gradient(Matrix $weights, Matrix $dOut) : Matrix
    {
        return $weights->transpose()->matmul($dOut);
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return \Generator<\Rubix\ML\NeuralNet\Parameter>
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
     * @param \Rubix\ML\NeuralNet\Parameter[] $parameters
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
