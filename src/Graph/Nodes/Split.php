<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;

/**
 * Split
 *
 * A split node that compares the values in a single feature column with the value on the node (i.e. the split value).
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Split implements Decision
{
    use HasBinaryChildren;

    /**
     * The feature column offset.
     *
     * @var int
     */
    protected $column;

    /**
     * The value to split on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right splits of the training data.
     *
     * @var array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    protected $groups;

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
     * @param int $column
     * @param string|int|float $value
     * @param array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled} $groups
     * @param float $impurity
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $column, $value, array $groups, float $impurity, int $n)
    {
        $this->column = $column;
        $this->value = $value;
        $this->groups = $groups;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the feature column (index) of the split value.
     *
     * @return int
     */
    public function column() : int
    {
        return $this->column;
    }

    /**
     * Return the split value.
     *
     * @return int|float|string
     */
    public function value()
    {
        return $this->value;
    }

    /**
     * Return the left and right splits of the training data.
     *
     * @return array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    public function groups() : array
    {
        return $this->groups;
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
     * Return the decrease in impurity this decision node introduces.
     *
     * @return float
     */
    public function purityIncrease() : float
    {
        $impurity = $this->impurity;

        if ($this->left instanceof Decision) {
            $impurity -= $this->left->impurity() * ($this->left->n() / $this->n);
        }

        if ($this->right instanceof Decision) {
            $impurity -= $this->right->impurity() * ($this->right->n() / $this->n);
        }

        return $impurity;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
