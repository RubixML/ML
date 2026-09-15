<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;

use const Rubix\ML\EPSILON;

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
 */
class Multiclass implements Output
{
    /**
     * The number of class neurons in the layer.
     *
     * @var int
     */
    protected int $numClasses;

    /**
     * The function that computes the loss of erroneous activations.
     *
     * @var ClassificationLoss
     */
    protected ClassificationLoss $costFn;

    /**
     * The memorized activation matrix.
     *
     * @var Matrix|null
     */
    protected ?Matrix $z = null;

    /**
     * Compute the Softmax activation.
     *
     * @param Matrix $x
     * @return Matrix
     */
    protected static function softmax(Matrix $x) : Matrix
    {
        $z = $x->transpose();

        $z = $z->subtractColumnVector($z->max())->exp();

        $total = $z->sum()->clipLower(EPSILON);

        return $z->divide($total)->transpose();
    }

    /**
     * @param int $numClasses
     * @param ClassificationLoss $costFn
     * @throws InvalidArgumentException
     */
    public function __construct(int $numClasses, ClassificationLoss $costFn)
    {
        if ($numClasses < 2) {
            throw new InvalidArgumentException('Number of classes'
                . " must be greater than 1, $numClasses given.");
        }

        if ($costFn instanceof BinaryCrossEntropy) {
            throw new InvalidArgumentException('Not compatible with binary cross entropy.');
        }

        $this->numClasses = $numClasses;
        $this->costFn = $costFn;
    }

    /**
     * Return the width of the layer.
     *
     * @return positive-int
     */
    public function width() : int
    {
        return $this->numClasses;
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
        $fanOut = $this->numClasses;

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
     * @param Matrix $x
     * @return Matrix
     */
    public function forward(Matrix $x) : Matrix
    {
        $z = self::softmax($x);

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
        return self::softmax($x);
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
        if (!$this->z) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $z = $this->z;

        $gradient = new Deferred([$this, 'gradient'], [$z, $y]);

        $loss = $this->costFn->compute($z, $y);

        $this->z = null;

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return Matrix
     */
    public function gradient(Matrix $z, Matrix $y) : Matrix
    {
        if ($this->costFn instanceof MulticlassCrossEntropy) {
            return $z->subtract($y)
                ->divide($z->n());
        }

        $dLoss = $this->costFn->differentiate($z, $y)
            ->divide($z->n());

        $zT = $z->transpose();

        $prod = $zT->multiply($dLoss->transpose());

        return $prod->subtract($zT->multiply($prod->sum()))->transpose();
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
