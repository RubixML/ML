<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use InvalidArgumentException;

use function gettype;

/**
 * Best
 *
 * A decision node whose outcome is the most probable class given a set
 * of class labels.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Best implements Outcome, Leaf
{
    use HasBinaryChildren;

    /**
     * The outcome of a decision.
     *
     * @var int|float|string
     */
    protected $outcome;

    /**
     * The probabilities of each discrete class outcome.
     *
     * @var float[]
     */
    protected $probabilities;

    /**
     * The amount of impurity among the labels in the node.
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
     * @param mixed $outcome
     * @param array $probabilities
     * @param float $impurity
     * @param int $n
     * @throws \InvalidArgumentException
     */
    public function __construct($outcome, array $probabilities, float $impurity, int $n)
    {
        if (!is_int($outcome) and !is_string($outcome)) {
            throw new InvalidArgumentException('Outcome must be an'
                . ' int or string, ' . gettype($outcome) . ' given.');
        }

        $this->outcome = $outcome;
        $this->probabilities = $probabilities;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the outcome of the decision.
     *
     * @return int|float|string
     */
    public function outcome()
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
}
