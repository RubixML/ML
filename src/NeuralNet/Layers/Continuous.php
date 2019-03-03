<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use InvalidArgumentException;
use RuntimeException;

/**
 * Continuous
 *
 * The Continuous output layer consists of a single linear neuron that outputs a
 * scalar value useful for regression problems.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Continuous implements Output
{
    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The function that computes the loss of bad activations.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFn;

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
     * The memoized output of the layer.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $z;

    /**
     * @param float $alpha
     * @param \Rubix\ML\NeuralNet\CostFunctions\CostFunction|null $costFn
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 1e-4, ?CostFunction $costFn = null)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization amount'
                . " must be 0 or greater, $alpha given.");
        }

        $this->alpha = $alpha;
        $this->costFn = $costFn ?? new LeastSquares();
        $this->initializer = new Xavier2();
    }

    /**
     * Return the width of the layer.
     *
     * @return int|null
     */
    public function width() : ?int
    {
        return 1;
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\NeuralNet\Parameter[]
     */
    public function parameters() : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
        }

        return [$this->weights, $this->biases];
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
        $fanOut = 1;

        $w = $this->initializer->initialize($fanIn, $fanOut);

        $this->weights = new Parameter($w);
        $this->biases = new Parameter(Matrix::zeros($fanOut, 1));

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
            throw new RuntimeException('Layer has not been initialized');
        }

        $this->input = $input;

        $this->z = $this->weights->w->matmul($input)
            ->add($this->biases->w->columnAsVector(0));

        return $this->z;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
        }

        return $this->weights->w->matmul($input)
            ->add($this->biases->w->columnAsVector(0));
    }

    /**
     * Calculate the gradients for each output neuron and update.
     *
     * @param array $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return array
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
        }

        if (!$this->input or !$this->z) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = Matrix::quick([$labels]);

        $delta = $this->costFn
            ->compute($expected, $this->z);

        $penalties = $this->weights->w->sum()
            ->multiply($this->alpha);

        $dL = $this->costFn
            ->differentiate($expected, $this->z, $delta)
            ->add($penalties)
            ->divide($this->z->n());

        $dW = $dL->matmul($this->input->transpose());
        $dB = $dL->sum()->asColumnMatrix();

        $w = $this->weights->w;

        $this->weights->w = $this->weights->w
            ->subtract($optimizer->step($this->weights, $dW));

        $this->biases->w = $this->biases->w
            ->subtract($optimizer->step($this->biases, $dB));

        $loss = $delta->sum()->mean();

        unset($this->input, $this->z);

        return [function () use ($w, $dL) {
            return $w->transpose()->matmul($dL);
        }, $loss];
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
            throw new RuntimeException('Layer has not been initialized');
        }

        return [
            'weights' => clone $this->weights->w,
            'biases' => clone $this->biases->w,
        ];
    }

    /**
     * Restore the parameters of the layer.
     *
     * @param array $parameters
     * @throws \RuntimeException
     */
    public function restore(array $parameters) : void
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
        }

        $this->weights->w = $parameters['weights'];
        $this->biases->w = $parameters['biases'];
    }
}
