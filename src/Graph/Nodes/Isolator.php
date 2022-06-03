<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use Traversable;

use function array_unique;
use function array_rand;
use function floor;
use function ceil;
use function min;
use function max;
use function getrandmax;
use function rand;

/**
 * Isolator
 *
 * Isolator nodes represent splits in a tree designed to isolate groups into cells by randomly
 * dividing them.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Isolator implements BinaryNode, HasBinaryChildren
{
    /**
     * The feature column (index) of the split value.
     *
     * @var int
     */
    protected int $column;

    /**
     * The value that the node splits on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right splits of the training data.
     *
     * @var list<\Rubix\ML\Datasets\Dataset>
     */
    protected array $groups;

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
     * Factory method to build a isolator node from a dataset using a random split of the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public static function split(Dataset $dataset) : self
    {
        $column = rand(0, $dataset->numFeatures() - 1);

        $values = $dataset->feature($column);

        $type = $dataset->featureType($column);

        if ($type->isContinuous()) {
            $min = min($values);
            $max = max($values);

            $phi = getrandmax() / max(abs($max), abs($min));

            $min = (int) floor($min * $phi);
            $max = (int) ceil($max * $phi);

            $value = rand($min, $max) / $phi;
        } else {
            $offset = array_rand(array_unique($values));

            $value = $values[$offset];
        }

        $groups = $dataset->splitByFeature($column, $value);

        return new self($column, $value, $groups);
    }

    /**
     * @param int $column
     * @param string|int|float $value
     * @param list<\Rubix\ML\Datasets\Dataset> $groups
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $column, $value, array $groups)
    {
        $this->column = $column;
        $this->value = $value;
        $this->groups = $groups;
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
     * @return list<\Rubix\ML\Datasets\Dataset>
     */
    public function groups() : array
    {
        return $this->groups;
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
