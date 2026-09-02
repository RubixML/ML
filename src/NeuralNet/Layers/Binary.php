<?php

namespace Rubix\ML\NeuralNet\Layers;

use NDArray;
use NumPower;
use Rubix\ML\Deferred;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
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

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('Number of classes must be 2, ' . count($classes) . ' given.');
        }

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
        if ($costFn instanceof MulticlassCrossEntropy) {
            throw new InvalidArgumentException('Not compatible with binary cross entropy.');
        }

        $classes = [
            $classes[0] => 0.0,
            $classes[1] => 1.0,
        ];

        $this->classes = $classes;
        $this->costFn = $costFn ?? new BinaryCrossEntropy();
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
        $output = $this->sigmoid->activate($input);

        $this->input = $input;
        $this->output = $output;

        return $output;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray
    {
        return $this->sigmoid->activate($input);
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param string[] $labels
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return (Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input or !$this->output) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $expected = [];

        foreach ($labels as $label) {
            $expected[] = $this->classes[$label];
        }

        $expected = NumPower::array([$expected], $this->input->dataType());

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
<<<<<<< HEAD
        $n = $output->shape()[1];

        // Optimization specific to sigmoid + binary cross entropy.
        // The loss derivative cancels with the sigmoid derivative, so dZ = (output - expected).
        if ($this->costFn instanceof BinaryCrossEntropy) {
            return NumPower::divide(
                NumPower::subtract($output, $expected),
                $n
            );
=======
        if ($this->costFn instanceof BinaryCrossEntropy) {
            return $output->subtract($expected)
                ->divide($output->n());
>>>>>>> dd5c3b6ec645db882b76156bd0c482ea1ebf53dd
        }

        $dLoss = NumPower::divide(
            $this->costFn->differentiate($output, $expected),
            $n
        );

        return NumPower::multiply(
            $this->sigmoid->differentiate($input, $output),
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
        return "Binary (cost function: {$this->costFn})";
    }
}
