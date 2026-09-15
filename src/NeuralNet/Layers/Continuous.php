<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\RegressionLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Continuous
 *
 * The Continuous output layer consists of a single linear neuron that outputs a scalar value.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Continuous implements Output
{
    /**
     * The function that computes the loss of erroneous activations.
     *
     * @var RegressionLoss
     */
    protected RegressionLoss $costFn;

    /**
     * The memorized input matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $x = null;

    /**
     * @param RegressionLoss|null $costFn
     */
    public function __construct(?RegressionLoss $costFn = null)
    {
        $this->costFn = $costFn ?? new LeastSquares();
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
     * @param Matrix $x
     * @return Matrix
     */
    public function forward(Matrix $x) : Matrix
    {
        $this->x = $x;

        return $x;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function infer(Matrix $x) : Matrix
    {
        return $x;
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param Matrix $y
     * @throws RuntimeException
     * @return (Deferred|float)[]
     */
    public function back(Matrix $y) : array
    {
        if (!$this->x) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $x = $this->x;

        $gradient = new Deferred([$this, 'gradient'], [$x, $y]);

        $loss = $this->costFn->compute($x, $y);

        $this->x = null;

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param Matrix $x
     * @param Matrix $expected
     * @return Matrix
     */
    public function gradient(Matrix $x, Matrix $expected) : Matrix
    {
        return $this->costFn->differentiate($x, $expected)
            ->divide($x->n());
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
        return "Continuous (cost function: {$this->costFn})";
    }
}
