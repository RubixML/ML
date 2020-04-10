<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use InvalidArgumentException;
use RuntimeException;
use Generator;

use function count;

/**
 * Binary
 *
 * This Binary layer consists of a single sigmoid neuron capable of
 * distinguishing between two discrete classes. The Binary layer is useful for
 * neural networks that output an either/or prediction such as yes or no.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Binary implements Output, Parametric
{
    /**
     * The labels of either of the possible outcomes.
     *
     * @var string[]
     */
    protected $classes = [
        //
    ];

    /**
     * The L2 regularization amount.
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
    protected $weightInitializer;

    /**
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $biasInitializer;

    /**
     * The function that outputs the activation or implulse of each neuron.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFn;

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
     * @var \Tensor\Matrix|null
     */
    protected $input;

    /**
     * The memoized z matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $z;

    /**
     * The memoized activation matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $computed;

    /**
     * @param string[] $classes
     * @param float $alpha
     * @param \Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss|null $costFn
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $weightInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $biasInitializer
     * @throws \InvalidArgumentException
     */
    public function __construct(
        array $classes,
        float $alpha = 1e-4,
        ?ClassificationLoss $costFn = null,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        $classes = array_unique($classes);

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be 2, ' . count($classes) . ' given.');
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " greater than 0, $alpha given.");
        }

        $this->classes = array_flip(array_values($classes));
        $this->alpha = $alpha;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->weightInitializer = $weightInitializer ?? new Xavier1();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.0);
        $this->activationFn = new Sigmoid();
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
            throw new RuntimeException('Layer has not been initialized');
        }

        $this->input = $input;

        $this->z = $this->weights->param()->matmul($input)
            ->add($this->biases->param());

        $this->computed = $this->activationFn->compute($this->z);

        return $this->computed;
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
            throw new RuntimeException('Layer has not been initialized.');
        }

        $z = $this->weights->param()->matmul($input)
            ->add($this->biases->param());

        return $this->activationFn->compute($z);
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param string[] $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return (\Rubix\ML\Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input or !$this->z or !$this->computed) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = [];

        foreach ($labels as $label) {
            $expected[] = $this->classes[$label];
        }

        $expected = Matrix::quick([$expected]);

        if ($this->costFn instanceof CrossEntropy) {
            $dA = $this->computed->subtract($expected)
                ->divide($this->computed->n());
        } else {
            $dL = $this->costFn->differentiate($this->computed, $expected)
                ->divide($this->computed->n());

            $dA = $this->activationFn->differentiate($this->z, $this->computed)
                ->multiply($dL);
        }

        $weights = $this->weights->param();

        $dW = $dA->matmul($this->input->transpose())
            ->add($weights->multiply($this->alpha));

        $dB = $dA->sum();

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        $gradient = new Deferred([$this, 'gradient'], [$weights, $dA]);

        $loss = $this->costFn->compute($this->computed, $expected);

        unset($this->input, $this->z, $this->computed);

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Tensor\Matrix $weights
     * @param \Tensor\Matrix $dA
     * @return \Tensor\Matrix
     */
    public function gradient(Matrix $weights, Matrix $dA) : Matrix
    {
        return $weights->transpose()->matmul($dA);
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
