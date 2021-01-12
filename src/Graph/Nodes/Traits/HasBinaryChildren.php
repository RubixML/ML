<?php

namespace Rubix\ML\Graph\Nodes\Traits;

use Rubix\ML\Graph\Nodes\BinaryNode;
use Traversable;

/**
 * Binary Children
 *
 * A node of a binary tree i.e a tree whose nodes have a maximum of two immediate children.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait HasBinaryChildren
{
    /**
     * The left child node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected $left;

    /**
     * The right child node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected $right;

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
    public function attachLeft(BinaryNode $node) : void
    {
        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     */
    public function attachRight(BinaryNode $node) : void
    {
        $this->right = $node;
    }

    /**
     * Detach the left child node.
     */
    public function detachLeft() : void
    {
        $this->left = null;
    }

    /**
     * Detach the right child node.
     */
    public function detachRight() : void
    {
        $this->right = null;
    }

    /**
     * Is this a leaf node? i.e no children.
     *
     * @return bool
     */
    public function leaf() : bool
    {
        return !$this->left and !$this->right;
    }
}
