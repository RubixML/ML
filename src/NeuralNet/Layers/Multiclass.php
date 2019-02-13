<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use InvalidArgumentException;
use RuntimeException;

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
    protected $costFunction;

    /**
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $initializer;

    /**
     * The function that outputs the activation or implulse of each neuron.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

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
     * @param \Rubix\ML\NeuralNet\CostFunctions\CostFunction $costFunction
     * @throws \InvalidArgumentException
     */
    public function __construct(array $classes, float $alpha = 1e-4, ?CostFunction $costFunction = null)
    {
        $classes = array_values(array_unique($classes));

        if (count($classes) < 2) {
            throw new InvalidArgumentException('The number of unique classes'
                . ' must be 2 or more, ' . count($classes) . ' given.');
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization amount'
                . " must be 0 or greater, $alpha given.");
        }

        if (is_null($costFunction)) {
            $costFunction = new CrossEntropy();
        }

        $this->classes = $classes;
        $this->alpha = $alpha;
        $this->costFunction = $costFunction;
        $this->initializer = new Xavier1();
        $this->activationFunction = new Softmax();
    }

    /**
     * Return the width of the layer.
     *
     * @return int|null
     */
    public function width() : ?int
    {
        return count($this->classes);
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
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = count($this->classes);

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
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        $this->input = $input;

        $this->z = $this->weights->w->matmul($input)
            ->add($this->biases->w->columnAsVector(0));

        $this->computed = $this->activationFunction->compute($this->z);

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
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        $z = $this->weights->w->matmul($input)
            ->add($this->biases->w->columnAsVector(0));

        return $this->activationFunction->compute($z);
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
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }

        if (is_null($this->input) or is_null($this->z) or is_null($this->computed)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = [];

        foreach ($this->classes as $i => $class) {
            $joint = [];

            foreach ($labels as $label) {
                $joint[] = $class === $label ? 1. : 0.;
            }

            $expected[] = $joint;
        }

        $expected = Matrix::quick($expected);

        $delta = $this->costFunction
            ->compute($expected, $this->computed);

        $penalties = $this->weights->w->sum()
            ->multiply($this->alpha);

        $dL = $this->costFunction
            ->differentiate($expected, $this->computed, $delta)
            ->add($penalties)
            ->divide($this->computed->n());

        $dA = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->multiply($dL);

        $dW = $dA->matmul($this->input->transpose());
        $dB = $dA->sum()->asColumnMatrix();

        $w = $this->weights->w;

        $this->weights->w = $this->weights->w
            ->subtract($optimizer->step($this->weights, $dW));

        $this->biases->w = $this->biases->w
            ->subtract($optimizer->step($this->biases, $dB));

        $loss = $delta->sum()->mean();

        unset($this->input, $this->z, $this->computed);

        return [function () use ($w, $dA) {
            return $w->transpose()->matmul($dA);
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
        if (is_null($this->weights) or is_null($this->biases)) {
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
        if (is_null($this->weights) or is_null($this->biases)) {
            throw new RuntimeException('Layer has not been initialized');
        }
        
        $this->weights->w = $parameters['weights'];
        $this->biases->w = $parameters['biases'];
    }
}
