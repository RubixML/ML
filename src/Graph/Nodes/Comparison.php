<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

/**
 * Comparison
 *
 * A decision node that marks a comparison between an input value and the
 * value of the comparison node.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Comparison extends Split implements Decision
{
    /**
     * The amount of impurity that the split introduces.
     *
     * @var float
     */
    protected $impurity;

    /**
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  int  $column
     * @param  mixed  $value
     * @param  array  $groups
     * @param  float  $impurity
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $column, $value, array $groups, float $impurity)
    {
        if ($impurity < 0.) {
            throw new InvalidArgumentException('Impurity cannot be less than'
                . " 0, $impurity given.");
        }

        $this->impurity = $impurity;
        $this->n = (int) array_sum(array_map('count', $groups));

        parent::__construct($column, $value, $groups);
    }

    /**
     * Return the impurity score of the node.
     *
     * @return float
     */
    public function impurity() : float
    {
        return $this->impurity;
    }

    /**
     * Return the number of samples from the training set this node represents.
     *
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }

    /**
     * Return the decrease in impurity this decision node introduces. A negative
     * impurity means that the decision node actually causes its children to become
     * less pure.
     *
     * @return float
     */
    public function purityIncrease() : float
    {
        $impurity = $this->impurity;

        if ($this->left instanceof Decision) {
            $impurity -= $this->left->impurity()
                * ($this->left->n() / $this->n);
        }

        if ($this->right instanceof Decision) {
            $impurity -= $this->right->impurity()
                * ($this->right->n() / $this->n);
        }

        return $impurity;
    }
}
