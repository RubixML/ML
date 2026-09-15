<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

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
    protected ?Matrix $x = null;

    /**
     * The memorized activation matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $z = null;

    /**
     * @param ClassificationLoss $costFn
     * @throws InvalidArgumentException
     */
    public function __construct(ClassificationLoss $costFn)
    {
        if ($costFn instanceof MulticlassCrossEntropy) {
            throw new InvalidArgumentException('Not compatible with multiclass cross entropy.');
        }

        $this->costFn = $costFn;
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
     * @param Matrix $x
     * @return Matrix
     */
    public function forward(Matrix $x) : Matrix
    {
        $z = $this->sigmoid->activate($x);

        $this->x = $x;
        $this->z = $z;

        return $z;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param Matrix $x
     * @return Matrix
     */
    public function infer(Matrix $x) : Matrix
    {
        return $this->sigmoid->activate($x);
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
        if (!$this->x or !$this->z) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $x = $this->x;
        $z = $this->z;

        $gradient = new Deferred([$this, 'gradient'], [$x, $z, $y]);

        $loss = $this->costFn->compute($z, $y);

        $this->x = $this->z = null;

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param Matrix $x
     * @param Matrix $z
     * @param Matrix $expected
     * @return Matrix
     */
    public function gradient(Matrix $x, Matrix $z, Matrix $expected) : Matrix
    {
        if ($this->costFn instanceof BinaryCrossEntropy) {
            return $z->subtract($expected)
                ->divide($z->n());
        }

        $dLoss = $this->costFn->differentiate($z, $expected)
            ->divide($z->n());

        return $this->sigmoid->differentiate($x, $z)
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
