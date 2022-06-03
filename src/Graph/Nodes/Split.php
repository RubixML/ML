<?php

namespace Rubix\ML\Graph\Nodes;

use Traversable;

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
class Split implements Decision, BinaryNode
{
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
     * The left and right splits of the training data.
     *
     * @var list<\Rubix\ML\Datasets\Labeled>
     */
    protected array $groups;

    /**
     * The amount of impurity that the split introduces.
     *
     * @var float
     */
    protected float $impurity;

    /**
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected int $n;

    /**
     * The left child node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected ?\Rubix\ML\Graph\Nodes\BinaryNode $left = null;

    /**
     * The right child node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected ?\Rubix\ML\Graph\Nodes\BinaryNode $right = null;

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
     * @return list<\Rubix\ML\Datasets\Labeled>
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
     * Return the left child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function left() : ?BinaryNode
    {
        return $this->left;
    }

    /**
     * Return the right child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function right() : ?BinaryNode
    {
        return $this->right;
    }

    /**
     * Return the children of this node in a generator.
     *
     * @return \Generator<\Rubix\ML\Graph\Nodes\BinaryNode>
     */
    public function children() : Traversable
    {
        if ($this->left) {
            yield $this->left;
        }

        if ($this->right) {
            yield $this->right;
        }
    }

    /**
     * Recursive function to determine the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return 1 + max($this->left ? $this->left->height() : 0, $this->right ? $this->right->height() : 0);
    }

    /**
     * The balance factor of the node. Negative numbers indicate a lean to the left, positive
     * to the right, and 0 is perfectly balanced.
     *
     * @return int
     */
    public function balance() : int
    {
        return ($this->right ? $this->right->height() : 0) - ($this->left ? $this->left->height() : 0);
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
     * Set the left child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     */
    public function attachLeft(?BinaryNode $node = null) : void
    {
        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     */
    public function attachRight(?BinaryNode $node = null) : void
    {
        $this->right = $node;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        $this->groups = [];
    }
}
