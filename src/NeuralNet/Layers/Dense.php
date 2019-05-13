<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use Rubix\ML\NeuralNet\Parameters\VectorParam;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use InvalidArgumentException;
use RuntimeException;
use Generator;

/**
 * Dense
 *
 * Dense layers are fully connected hidden layers, meaning each neuron is
 * connected to each other neuron in the previous layer by a weighted *synapse*.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dense implements Hidden, Parametric
{
    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int
     */
    protected $neurons;

    /**
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $weightInitializer;

    /**
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $biasInitializer;

    /**
     * The weights.
     *
     * @var \Rubix\ML\NeuralNet\Parameters\Parameter|null
     */
    protected $weights;

    /**
     * The biases.
     *
     * @var \Rubix\ML\NeuralNet\Parameters\Parameter|null
     */
    protected $biases;

    /**
     * The memoized input matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $input;

    /**
     * @param int $neurons
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $weightInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $biasInitializer
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $neurons,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons cannot be'
                . ' less than 1.');
        }

        $this->neurons = $neurons;
        $this->weightInitializer = $weightInitializer ?? new He();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.);
    }

    /**
     * Return the width of the layer.
     *
     * @return int
     */
    public function width() : int
    {
        return $this->neurons;
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator
     */
    public function parameters() : Generator
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        yield $this->weights;
        yield $this->biases;
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
        $fanOut = $this->neurons;

        $w = $this->weightInitializer->initialize($fanIn, $fanOut);
        $b = $this->biasInitializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->weights = new MatrixParam($w);
        $this->biases = new VectorParam($b);

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        $this->input = $input;

        return $this->weights->w()->matmul($input)
            ->add($this->biases->w());
    }

    /**
     * Compute an inference pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        return $this->weights->w()->matmul($input)
            ->add($this->biases->w());
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
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient->compute();

        $dW = $dOut->matmul($this->input->transpose());
        $dB = $dOut->sum();

        $w = $this->weights->w();

        $optimizer->step($this->weights, $dW);
        $optimizer->step($this->biases, $dB);

        unset($this->input);

        return new Deferred([$this, 'gradient'], [$w, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Rubix\Tensor\Matrix $w
     * @param \Rubix\Tensor\Matrix $dOut
     * @return \Rubix\Tensor\Matrix
     */
    public function gradient(Matrix $w, Matrix $dOut) : Matrix
    {
        return $w->transpose()->matmul($dOut);
    }

    /**
     * Return the parameters of the layer in an associative array.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function read() : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        return [
            'weights' => clone $this->weights,
            'biases' => clone $this->biases,
        ];
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param array $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'];
    }
}
