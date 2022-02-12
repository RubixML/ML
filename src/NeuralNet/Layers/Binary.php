<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;

/**
 * Binary
 *
 * This Binary layer consists of a single sigmoid neuron capable of distinguishing between
 * two discrete classes.
 *
 * @internal
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
     * The function that computes the loss of erroneous activations.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFn;

    /**
     * The sigmoid activation function.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid
     */
    protected $activationFn;

    /**
     * The memorized input matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $input;

    /**
     * The memorized activation matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $computed;

    /**
     * @param string[] $classes
     * @param \Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss|null $costFn
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $classes, ?ClassificationLoss $costFn = null)
    {
        $classes = array_unique($classes);

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be 2, ' . count($classes) . ' given.');
        }

        $this->classes = array_map('strval', array_flip(array_values($classes)));
        $this->costFn = $costFn ?? new CrossEntropy();
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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        if ($fanIn !== 1) {
            throw new InvalidArgumentException('Fan in must be'
                . " equal to 1, $fanIn given.");
        }

        return 1;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        $this->computed = $this->activationFn->compute($input);

        return $this->computed;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->activationFn->compute($input);
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param string[] $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return (\Rubix\ML\Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input or !$this->computed) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $expected = [];

        foreach ($labels as $label) {
            $expected[] = $this->classes[$label];
        }

        $expected = Matrix::quick([$expected]);

        $input = $this->input;
        $computed = $this->computed;

        $gradient = new Deferred([$this, 'gradient'], [$input, $computed, $expected]);

        $loss = $this->costFn->compute($computed, $expected);

        $this->input = $this->computed = null;

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $computed
     * @param \Tensor\Matrix $expected
     * @return \Tensor\Matrix
     */
    public function gradient(Matrix $input, Matrix $computed, Matrix $expected) : Matrix
    {
        if ($this->costFn instanceof CrossEntropy) {
            return $computed->subtract($expected)
                ->divide($computed->n());
        }

        $dL = $this->costFn->differentiate($computed, $expected)
            ->divide($computed->n());

        return $this->activationFn->differentiate($input, $computed)
            ->multiply($dL);
    }
}
