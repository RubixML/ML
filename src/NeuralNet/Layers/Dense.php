<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use InvalidArgumentException;
use RuntimeException;

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
    protected $initializer;

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
     * The memoized input matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $input;

    /**
     * @param  int  $neurons
     * @param  \Rubix\ML\NeuralNet\Initializers\Initializer  $initializer
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, Initializer $initializer = null)
    {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons cannot be'
                . ' less than 1.');
        }

        if (is_null($initializer)) {
            $initializer = new He();
        }

        $this->neurons = $neurons;
        $this->initializer = $initializer;
    }

    /**
     * Return the width of the layer.
     * 
     * @return int|null
     */
    public function width() : ?int
    {
        return $this->neurons;
    }

    /**
     * Return the parameters of the layer.
     * 
     * @throws \RuntimeException
     * @return \Rubix\ML\NeuralNet\Parameter[]
     */
    public function parameters() : array
    {
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        return [$this->weights, $this->biases];
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $this->neurons;

        $w = $this->initializer->initialize($fanIn, $fanOut);

        $this->weights = new Parameter($w);
        $this->biases = new Parameter(Matrix::zeros($fanOut, 1));

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        $this->input = $input;

        return $this->weights->w->matmul($input)
            ->add($this->biases->w->columnAsVector(0));
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        return $this->weights->w->matmul($input)
            ->add($this->biases->w->columnAsVector(0));
    }

    /**
     * Calculate the gradients and update the parameters of the layer.
     *
     * @param  callable  $prevGradient
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return callable
     */
    public function back(callable $prevGradient, Optimizer $optimizer) : callable
    {
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        if (is_null($this->input)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient();

        $dW = $dOut->matmul($this->input->transpose());
        $dB = $dOut->sum()->asColumnMatrix();

        $w = $this->weights->w;

        $this->weights->w = $this->weights->w
            ->subtract($optimizer->step($this->weights, $dW));

        $this->biases->w = $this->biases->w
            ->subtract($optimizer->step($this->biases, $dB));

        unset($this->input);

        return function () use ($w, $dOut) {
            return $w->transpose()->matmul($dOut);
        };
    }

    /**
     * Return the parameters of the layer in an associative array.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function read() : array
    {
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        return [
            'weights' => clone $this->weights->w,
            'biases' => clone $this->biases->w,
        ];
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param  array  $parameters
     * @throws \RuntimeException
     * @return void
     */
    public function restore(array $parameters) : void
    {
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }
        
        $this->weights->w = $parameters['weights'];
        $this->biases->w = $parameters['biases'];
    }
}
