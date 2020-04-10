<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\RegressionLoss;
use InvalidArgumentException;
use RuntimeException;
use Generator;

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
class Continuous implements Output, Parametric
{
    /**
     * The amount of L2 regularization applied to the weights.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The function that computes the loss of bad activations.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\RegressionLoss
     */
    protected $costFn;

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
     * The memorized input matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $input;

    /**
     * The memorized output of the layer.
     *
     * @var \Tensor\Matrix|null
     */
    protected $z;

    /**
     * @param float $alpha
     * @param \Rubix\ML\NeuralNet\CostFunctions\RegressionLoss|null $costFn
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $weightInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $biasInitializer
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $alpha = 0.0,
        ?RegressionLoss $costFn = null,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " greater than 0, $alpha given.");
        }

        $this->alpha = $alpha;
        $this->costFn = $costFn ?? new LeastSquares();
        $this->weightInitializer = $weightInitializer ?? new Xavier2();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.0);
    }

    /**
     * Return the width of the layer.
     *
     * @return int
     */
    public function width() : int
    {
        return 1;
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

        $weights = $this->weightInitializer->initialize($fanIn, $fanOut);
        $biases = $this->biasInitializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->weights = new Parameter($weights);
        $this->biases = new Parameter($biases);

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $this->input = $input;

        $this->z = $this->weights->param()->matmul($input)
            ->add($this->biases->param());

        return $this->z;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        return $this->weights->param()->matmul($input)
            ->add($this->biases->param());
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param (int|float)[] $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return (\Rubix\ML\Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer is not initialized');
        }

        if (!$this->input or !$this->z) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = Matrix::quick([$labels]);

        $dL = $this->costFn->differentiate($this->z, $expected)
            ->divide($this->z->n());

        $dW = $dL->matmul($this->input->transpose());
        $dB = $dL->sum();

        $weights = $this->weights->param();

        if ($this->alpha) {
            $dW = $dW->add($weights->multiply($this->alpha));
        }

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        $gradient = new Deferred([$this, 'gradient'], [$weights, $dL]);

        $loss = $this->costFn->compute($this->z, $expected);

        unset($this->input, $this->z);

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Tensor\Matrix $weights
     * @param \Tensor\Matrix $dL
     * @return \Tensor\Matrix
     */
    public function gradient(Matrix $weights, Matrix $dL) : Matrix
    {
        return $weights->transpose()->matmul($dL);
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator<\Rubix\ML\NeuralNet\Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'weights' => $this->weights;
        yield 'biases' => $this->biases;
    }

    /**
     * Restore the parameters of the layer.
     *
     * @param \Rubix\ML\NeuralNet\Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'];
    }
}
