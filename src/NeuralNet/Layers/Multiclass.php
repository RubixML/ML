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
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\Exceptions\RuntimeException;

use function count;

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
    public function __construct(array $classes, ClassificationLoss $costFn)
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

        if ($costFn instanceof BinaryCrossEntropy) {
            throw new InvalidArgumentException('Not compatible with binary cross entropy.');
        }

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
     * @param string $dataType
     * @throws InvalidArgumentException
     * @return positive-int
     */
    public function initialize(int $fanIn, string $dataType) : int
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

        $expected = NumPower::array($expected, $this->input->dataType());

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
     * For Cross Entropy, the loss derivative cancels with the Softmax
     * derivative, so dZ = (output - expected). Otherwise, an exact backward
     * pass through Softmax is computed as the Jacobian-vector product
     *
     * dZ_i = s_i * (g_i - sum_j(s_j * g_j))
     *
     * where s is the Softmax output and g is the upstream gradient.
     *
     * @param NDArray $input
     * @param NDArray $output
     * @param NDArray $expected
     * @return NDArray
     */
    public function gradient(NDArray $input, NDArray $output, NDArray $expected) : NDArray
    {
<<<<<<< HEAD
        $n = array_product($output->shape());

        // Optimization specific to softmax + multiclass cross entropy.
        // The loss derivative cancels with the softmax derivative, so dZ = (output - expected).
        if ($this->costFn instanceof MulticlassCrossEntropy) {
            return NumPower::divide(
                NumPower::subtract($output, $expected),
                $n
            );
=======
        if ($this->costFn instanceof MulticlassCrossEntropy) {
            return $output->subtract($expected)
                ->divide($output->n());
>>>>>>> dd5c3b6ec645db882b76156bd0c482ea1ebf53dd
        }

        $gradient = NumPower::divide(
            $this->costFn->differentiate($output, $expected),
            $n
        );

        $batch = $output->shape()[1];

        $dot = NumPower::reshape(
            NumPower::sum(NumPower::multiply($gradient, $output), axis: 0),
            [1, $batch]
        );

        return NumPower::multiply(
            $output,
            NumPower::subtract($gradient, $dot)
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
