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
     * @var float[]
     */
    protected array $classes = [
        //
    ];

    /**
     * The function that computes the loss of erroneous activations.
     *
     * @var ClassificationLoss
     */
    protected ClassificationLoss $costFn;

    /**
     * The sigmoid activation function.
     *
     * @var Sigmoid
     */
    protected Sigmoid $sigmoid;

    /**
     * The memorized input matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $input = null;

    /**
     * The memorized activation matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $output = null;

    /**
     * @param string[] $classes
     * @param ClassificationLoss|null $costFn
     * @throws InvalidArgumentException
     */
    public function __construct(array $classes, ?ClassificationLoss $costFn = null)
    {
        $classes = array_values(array_unique($classes));

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be 2, ' . count($classes) . ' given.');
        }

        $classes = [
            $classes[0] => 0.0,
            $classes[1] => 1.0,
        ];

        $this->classes = $classes;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->sigmoid = new Sigmoid();
    }

    /**
     * Return the width of the layer.
     *
     * @return positive-int
     */
    public function width() : int
    {
        return 1;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param positive-int $fanIn
     * @throws InvalidArgumentException
     * @return positive-int
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
     * @param Matrix $input
     * @return Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $output = $this->sigmoid->activate($input);

        $this->input = $input;
        $this->output = $output;

        return $output;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->sigmoid->activate($input);
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param string[] $labels
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return (\Rubix\ML\Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input or !$this->output) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $expected = [];

        foreach ($labels as $label) {
            $expected[] = $this->classes[$label];
        }

        $expected = Matrix::quick([$expected]);

        $input = $this->input;
        $output = $this->output;

        $gradient = new Deferred([$this, 'gradient'], [$input, $output, $expected]);

        $loss = $this->costFn->compute($output, $expected);

        $this->input = $this->output = null;

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param Matrix $input
     * @param Matrix $output
     * @param Matrix $expected
     * @return Matrix
     */
    public function gradient(Matrix $input, Matrix $output, Matrix $expected) : Matrix
    {
        if ($this->costFn instanceof CrossEntropy) {
            return $output->subtract($expected)
                ->divide($output->n());
        }

        $dLoss = $this->costFn->differentiate($output, $expected)
            ->divide($output->n());

        return $this->sigmoid->differentiate($input, $output)
            ->multiply($dLoss);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Binary (cost function: {$this->costFn})";
    }
}
