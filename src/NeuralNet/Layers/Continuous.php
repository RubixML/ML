<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
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
     * @var NDArray|null
     */
    protected ?NDArray $input = null;

    /**
     * @param RegressionLoss|null $costFn
     */
    public function __construct(?RegressionLoss $costFn = null)
    {
        $this->costFn = $costFn ?? new LeastSquares();

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
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
     * @param string $dataType
     * @throws InvalidArgumentException
     * @return positive-int
     */
    public function initialize(int $fanIn, string $dataType) : int
    {
        if ($fanIn !== 1) {
            throw new InvalidArgumentException("Fan in must be equal to 1, $fanIn given.");
        }

        return 1;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray
    {
        $this->input = $input;

        return $input;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        return $input;
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param (int|float)[] $labels
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return (Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $expected = NumPower::array([$labels], $this->input->dataType());

        $input = $this->input;

        $gradient = new Deferred([$this, 'gradient'], [$input, $expected]);

        $loss = $this->costFn->compute($input, $expected);

        $this->input = null;

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param NDArray $input
     * @param NDArray $expected
     * @return NDArray
     */
    public function gradient(NDArray $input, NDArray $expected) : NDArray
    {
        $n = $input->shape()[1];

        return NumPower::divide(
            $this->costFn->differentiate($input, $expected),
            $n
        );
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
