<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\Backends\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use Rubix\ML\NeuralNet\Parameters\VectorParam;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use InvalidArgumentException;
use RuntimeException;
use Generator;

/**
 * Multiclass
 *
 * The Multiclass output layer gives a joint probability estimate of a multiclass
 * classification problem using the Softmax activation function.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Multiclass implements Output
{
    /**
     * The unique class labels.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

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
     * The memoized z matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $z;

    /**
     * The memoized activation matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $computed;

    /**
     * @param array $classes
     * @param float $alpha
     * @param \Rubix\ML\NeuralNet\CostFunctions\CostFunction|null $costFn
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $weightInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $biasInitializer
     * @throws \InvalidArgumentException
     */
    public function __construct(
        array $classes,
        float $alpha = 1e-4,
        ?CostFunction $costFn = null,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        $classes = array_values(array_unique($classes));

        if (count($classes) < 2) {
            throw new InvalidArgumentException('The number of unique classes'
                . ' must be 2 or more, ' . count($classes) . ' given.');
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization coefficient'
                . " must be 0 or greater, $alpha given.");
        }

        $this->classes = $classes;
        $this->alpha = $alpha;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->weightInitializer = $weightInitializer ?? new Xavier1();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.);
        $this->activationFn = new Softmax();
    }

    /**
     * Return the width of the layer.
     *
     * @return int
     */
    public function width() : int
    {
        return count($this->classes);
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
        $fanOut = count($this->classes);

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

        $this->z = $this->weights->w()->matmul($input)
            ->add($this->biases->w());

        $this->computed = $this->activationFn->compute($this->z);

        return $this->computed;
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
            throw new RuntimeException('Layer is not initialized');
        }

        $z = $this->weights->w()->matmul($input)
            ->add($this->biases->w());

        return $this->activationFn->compute($z);
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
            throw new RuntimeException('Layer is not initialized');
        }

        if (!$this->input or !$this->z or !$this->computed) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = [];

        foreach ($this->classes as $class) {
            $temp = [];

            foreach ($labels as $label) {
                $temp[] = $class === $label ? 1. : 0.;
            }

            $expected[] = $temp;
        }

        $expected = Matrix::quick($expected);

        $penalties = $this->weights->w()->sum()
            ->multiply($this->alpha);

        if ($this->costFn instanceof CrossEntropy) {
            $dA = $this->computed
                ->subtract($expected)
                ->add($penalties)
                ->divide($this->computed->n());
        } else {
            $dL = $this->costFn
                ->differentiate($expected, $this->computed)
                ->add($penalties)
                ->divide($this->computed->n());

            $dA = $this->activationFn
                ->differentiate($this->z, $this->computed)
                ->multiply($dL);
        }

        $dW = $dA->matmul($this->input->transpose());
        $dB = $dA->sum();

        $w = $this->weights->w();

        $optimizer->step($this->weights, $dW);
        $optimizer->step($this->biases, $dB);

        $loss = $this->costFn->compute($expected, $this->computed);

        $gradient = new Deferred(function () use ($w, $dA) {
            return $w->transpose()->matmul($dA);
        });

        unset($this->input, $this->z, $this->computed);

        return [$gradient, $loss];
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
     * Restore the parameters of the layer.
     *
     * @param array $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'];
    }
}
