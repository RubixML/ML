<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildrenTrait;
use Rubix\ML\Exceptions\RuntimeException;

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
class Split implements Decision, HasBinaryChildren
{
    use HasBinaryChildrenTrait;

    /**
     * The feature column offset.
     *
     * @var int
     */
    protected int $column;

    /**
     * The value to split on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right subsets of the training data.
     *
     * @var array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    protected array $subsets;

    /**
     * The amount of impurity that the split introduces.
     *
     * @var float
     */
    protected float $impurity;

    /**
     * The number of training samples this node is responsible for.
     *
     * @var int<0,max>
     */
    protected int $n;

    /**
     * @param int $column
     * @param string|int|float $value
     * @param array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled} $subsets
     * @param float $impurity
     * @param int<0,max> $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $column, $value, array $subsets, float $impurity, int $n)
    {
        $this->column = $column;
        $this->value = $value;
        $this->subsets = $subsets;
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
     * Return the left and right subsets of the training data.
     *
     * @throws RuntimeException
     * @return array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    public function subsets() : array
    {
        if (!isset($this->subsets)) {
            throw new RuntimeException('Subsets property does not exist.');
        }

        return $this->subsets;
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
     * @return int<0,max>
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
     * Remove any variables carried over from the parent node.
     */
    public function cleanup() : void
    {
        unset($this->subsets);
    }
}
