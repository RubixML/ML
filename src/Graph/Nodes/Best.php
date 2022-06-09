<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Best
 *
 * A decision node whose outcome is the most probable class given a set of class labels.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Best implements Outcome
{
    /**
     * The outcome of a decision.
     *
     * @var string
     */
    protected string $outcome;

    /**
     * The probabilities of each discrete class outcome.
     *
     * @var float[]
     */
    protected array $probabilities;

    /**
     * The amount of impurity among the labels in the node.
     *
     * @var float
     */
    protected float $impurity;

    /**
     * The number of labels this node is responsible for.
     *
     * @var int
     */
    protected int $n;

    /**
     * @param string $outcome
     * @param (int|float)[] $probabilities
     * @param float $impurity
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(string $outcome, array $probabilities, float $impurity, int $n)
    {
        $this->outcome = $outcome;
        $this->probabilities = $probabilities;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the outcome of the decision.
     *
     * @return string
     */
    public function outcome() : string
    {
        return $this->outcome;
    }

    /**
     * Return the probabilities of each possible outcome.
     *
     * @return float[]
     */
    public function probabilities() : array
    {
        return $this->probabilities;
    }

    /**
     * Return the impurity within the node.
     *
     * @return float
     */
    public function impurity() : float
    {
        return $this->impurity;
    }

    /**
     * Return the number of labels within the node.
     *
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }

    /**
     * Return the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return 1;
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
        return "Best {outcome: {$this->outcome}, impurity: {$this->impurity}, n: {$this->n}}";
    }
}
