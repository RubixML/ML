<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

/**
 * Outcome
 *
 * A decision node whose outcome is the most probable class given a set
 * of class labels.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Outcome extends BinaryNode implements Purity, Leaf
{
    /**
     * The class outcome of a decision.
     *
     * @var int|string
     */
    protected $class;

    /**
     * The probabilities of each discrete class outcome.
     *
     * @var float[]
     */
    protected $probabilities;

    /**
     * The amount of impurity within the labels of the node.
     *
     * @var float
     */
    protected $impurity;

    /**
     * The number of labels this node is responsible for.
     *
     * @var int
     */
    protected $n;
    
    /**
     * @param mixed $class
     * @param array $probabilities
     * @param float $impurity
     * @param int $n
     * @throws \InvalidArgumentException
     */
    public function __construct($class, array $probabilities, float $impurity, int $n)
    {
        if (!is_int($class) and !is_string($class)) {
            throw new InvalidArgumentException('Class outcome must be an'
                . ' integer or string, ' . gettype($class) . ' given.');
        }

        $this->class = $class;
        $this->probabilities = $probabilities;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the outcome of the decision i.e the most probable outcome.
     *
     * @return int|string
     */
    public function class()
    {
        return $this->class;
    }

    /**
     * Return the proababilities of each discrete outcome.
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
}
