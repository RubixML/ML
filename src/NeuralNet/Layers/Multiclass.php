<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Multiclass
 *
 * The Multiclass output layer gives a joint probability estimate of a multiclass classification
 * problem using the Softmax activation function.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Multiclass implements Output
{
    /**
     * The unique class labels.
     *
     * @var string[]
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
     * The softmax activation function.
     *
     * @var Softmax
     */
    protected Softmax $softmax;

    /**
     * The memorized input matrix.
     *
     * @var NDArray|null
     */
    protected ?NDArray $input = null;

    /**
     * The memorized activation matrix.
     *
     * @var NDArray|null
     */
    protected ?NDArray $output = null;

    /**
     * @param string[] $classes
     * @param ClassificationLoss|null $costFn
     * @throws InvalidArgumentException
     */
    public function __construct(array $classes, ?ClassificationLoss $costFn = null)
    {
        $classes = array_values(array_unique($classes));

        if (count($classes) < 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be greater than 1, ' . count($classes)
                . ' given.');
        }

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->classes = $classes;
        $this->costFn = $costFn ?? new MulticlassCrossEntropy();
        $this->softmax = new Softmax();
    }

    /**
     * Return the width of the layer.
     *
     * @return positive-int
     */
    public function width() : int
    {
        return max(1, count($this->classes));
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
        $fanOut = count($this->classes);

        if ($fanIn !== $fanOut) {
            throw new InvalidArgumentException('Fan in must be'
                . " equal to fan out, $fanOut expected but"
                . " $fanIn given.");
        }

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray
    {
        $output = $this->softmax->activate($input);

        $this->input = $input;
        $this->output = $output;

        return $output;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param NDArray $input
     * @throws RuntimeException
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        return $this->softmax->activate($input);
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param string[] $labels
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return array{0: Deferred, 1: float}
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input or !$this->output) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        // Build one-hot targets as [classes, batch] to match Dense output layout.
        $expected = [];

        foreach ($this->classes as $class) {
            $row = [];

            foreach ($labels as $label) {
                $row[] = $class == $label ? 1.0 : 0.0;
            }

            $expected[] = $row;
        }

        $expected = NumPower::array($expected, 'float32');

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
     * @param NDArray $input
     * @param NDArray $output
     * @param NDArray $expected
     * @return NDArray
     */
    public function gradient(NDArray $input, NDArray $output, NDArray $expected) : NDArray
    {
        $n = array_product($output->shape());

        if ($this->costFn instanceof MulticlassCrossEntropy) {
            return NumPower::divide(
                NumPower::subtract($output, $expected),
                $n
            );
        }

        $dLoss = NumPower::divide(
            $this->costFn->differentiate($output, $expected),
            $n
        );

        return NumPower::multiply(
            $this->softmax->differentiate($input, $output),
            $dLoss
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
        return "Multiclass (cost function: {$this->costFn})";
    }
}
