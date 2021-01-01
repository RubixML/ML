<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Helpers\Params;
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
     * @var int
     */
    protected $neurons;

    /**
     * The amount of L2 regularization applied to the weights.
     *
     * @var float
     */
    protected $alpha;

    /**
     * Should the layer include a bias parameter?
     *
     * @var bool
     */
    protected $bias;

    /**
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $weightInitializer;

    /**
     * The bias initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $biasInitializer;

    /**
     * The weights.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $weights;

    /**
     * The biases.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $biases;

    /**
     * The memorized inputs to the layer.
     *
     * @var \Tensor\Matrix|null
     */
    protected $input;

    /**
     * @param int $neurons
     * @param float $alpha
     * @param bool $bias
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $weightInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $biasInitializer
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        int $neurons,
        float $alpha = 0.0,
        bool $bias = true,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        if ($neurons < 1) {
            throw new InvalidArgumentException('Number of neurons'
                . " must be greater than 0, $neurons given.");
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " greater than 0, $alpha given.");
        }

        $this->neurons = $neurons;
        $this->alpha = $alpha;
        $this->bias = $bias;
        $this->weightInitializer = $weightInitializer ?? new He();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.0);
    }

    /**
     * Return the width of the layer.
     *
     * @internal
     *
     * @return int
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
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
     * @param int $fanIn
     * @return int
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
     * @param \Tensor\Matrix $input
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $z = $this->weights->param()->matmul($input);

        if ($this->biases) {
            $z = $z->add($this->biases->param());
        }

        $this->input = $input;

        return $z;
    }

    /**
     * Compute an inference pass through the layer.
     *
     * @internal
     *
     * @param \Tensor\Matrix $input
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $z = $this->weights->param()->matmul($input);

        if ($this->biases) {
            $z = $z->add($this->biases->param());
        }

        return $z;
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

        if ($this->alpha) {
            $dW = $dW->add($weights->multiply($this->alpha));
        }

        $this->weights->update($optimizer->step($this->weights, $dW));

        if ($this->biases) {
            $dB = $dOut->sum();

            $this->biases->update($optimizer->step($this->biases, $dB));
        }

        $this->input = null;

        return new Deferred([$this, 'gradient'], [$weights, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param \Tensor\Matrix $weights
     * @param \Tensor\Matrix $dOut
     * @return \Tensor\Matrix
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
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
     * @return string
     */
    public function __toString() : string
    {
        return "Dense (neurons: {$this->neurons}, alpha: {$this->alpha},"
            . ' bias: ' . Params::toString($this->bias) . ','
            . " weight_initializer: {$this->weightInitializer},"
            . " bias_initializer: {$this->biasInitializer})";
    }
}
