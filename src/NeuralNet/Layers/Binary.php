<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use Rubix\ML\NeuralNet\Parameters\VectorParam;
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
class Binary implements Output
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
            throw new InvalidArgumentException('The number of unique classes'
                . ' must be exactly 2.');
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization amount'
                . " must be 0 or greater, $alpha given.");
        }

        $this->classes = array_flip(array_values($classes));
        $this->alpha = $alpha;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->weightInitializer = $weightInitializer ?? new Xavier1();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.);
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
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator<\Rubix\ML\NeuralNet\Parameters\Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
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
        $fanOut = 1;

        $w = $this->weightInitializer->initialize($fanIn, $fanOut);

        $b = $this->biasInitializer->initialize(1, $fanOut)
            ->columnAsVector(0);

        $this->weights = new MatrixParam($w);
        $this->biases = new VectorParam($b);

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

        $this->z = $this->weights->w()->matmul($input)
            ->add($this->biases->w());

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
            throw new RuntimeException('Layer has not been initialized');
        }

        $z = $this->weights->w()->matmul($input)
            ->add($this->biases->w());

        return $this->activationFn->compute($z);
    }

    /**
     * Calculate the gradients for each output neuron and update.
     *
     * @param string[] $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return mixed[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
        }

        if (!$this->input or !$this->z or !$this->computed) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = [];

        foreach ($labels as $label) {
            $expected[] = $this->classes[$label];
        }

        $target = Matrix::quick([$expected]);

        $dPenalties = $this->weights->w()->sum()
            ->multiply($this->alpha);

        if ($this->costFn instanceof CrossEntropy) {
            $dA = $this->computed
                ->subtract($target)
                ->add($dPenalties)
                ->divide($this->computed->n());
        } else {
            $dL = $this->costFn
                ->differentiate($this->computed, $target)
                ->add($dPenalties)
                ->divide($this->computed->n());

            $dA = $this->activationFn
                ->differentiate($this->z, $this->computed)
                ->multiply($dL);
        }

        $dW = $dA->matmul($this->input->transpose());
        $dB = $dA->sum();

        $w = $this->weights->w();

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        $gradient = new Deferred([$this, 'gradient'], [$w, $dA]);

        $loss = $this->costFn->compute($this->computed, $target);

        unset($this->input, $this->z, $this->computed);

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Tensor\Matrix $w
     * @param \Tensor\Matrix $dA
     * @return \Tensor\Matrix
     */
    public function gradient(Matrix $w, Matrix $dA) : Matrix
    {
        return $w->transpose()->matmul($dA);
    }

    /**
     * Return the parameters of the layer in an associative array.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\NeuralNet\Parameters\Parameter[]
     */
    public function read() : array
    {
        if (!$this->weights or !$this->biases) {
            throw new RuntimeException('Layer has not been initialized');
        }

        return [
            'weights' => clone $this->weights,
            'biases' => clone $this->biases,
        ];
    }

    /**
     * Restore the parameters of the layer.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'];
    }
}
