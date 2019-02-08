<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Helpers\Stats;
use InvalidArgumentException;

/**
 * Average
 *
 * A decision node whose outcome is the average of all the labels it is
 * responsible for.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Average extends BinaryNode implements Decision, Leaf
{
    /**
     * The outcome of the decision as an average of the labels
     * contained within.
     *
     * @var int|float
     */
    protected $outcome;

    /**
     * The amount of impurity within the node.
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
     * @param float $impurity
     * @param int $n
     * @throws \InvalidArgumentException
     */
    public function __construct($outcome, float $impurity, int $n)
    {
        if (!is_int($outcome) and !is_float($outcome)) {
            throw new InvalidArgumentException('Outcome must be an'
                . ' integer or float, ' . gettype($outcome)
                . ' found.');
        }

        $this->outcome = $outcome;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the outcome of the decision i.e the average of the
     * labels.
     *
     * @return int|float
     */
    public function outcome()
    {
        return $this->outcome;
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
