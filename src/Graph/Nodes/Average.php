<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;

/**
 * Average
 *
 * A decision node whose outcome is the average of all the labels it is responsible for.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Average implements Outcome
{
    use HasBinaryChildren;

    /**
     * The average of the labels contained within.
     *
     * @var int|float
     */
    protected $outcome;

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
     * @param int|float $outcome
     * @param float $impurity
     * @param int $n
     */
    public function __construct($outcome, float $impurity, int $n)
    {
        $this->outcome = $outcome;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the outcome of the decision i.e the average of the labels.
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

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Average {outcome: {$this->outcome}, impurity: {$this->impurity}, n: {$this->n}}";
    }
}
