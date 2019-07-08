<?php

namespace Rubix\ML\Graph\Nodes;

use Generator;

/**
 * Binary Node
 *
 * A node of a binary tree i.e a tree whose nodes have a maximum of
 * two immediate children and one parent.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BinaryNode implements Node
{
    /**
     * The parent node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected $parent;

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
     * Return the parent node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function parent() : ?self
    {
        return $this->parent;
    }

    /**
     * Return the children of this node in a generator.
     *
     * @return \Generator
     */
    public function children() : Generator
    {
        if ($this->left) {
            yield $this->left;
        }
        
        if ($this->right) {
            yield $this->right;
        }
    }

    /**
     * Return a generator for all of the node's edges i.e. the nodes that
     * this node connects to.
     *
     * @return Generator
     */
    public function edges() : Generator
    {
        foreach ($this->children() as $node) {
            yield $node;
        }

        if ($this->parent) {
            yield $this->parent;
        }
    }

    /**
     * Return the left child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function left() : ?self
    {
        return $this->left;
    }

    /**
     * Return the right child node.
     *
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function right() : ?self
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
        return 1 + max(
            $this->left ? $this->left->height() : 0,
            $this->right ? $this->right->height() : 0
        );
    }

    /**
     * The balance factor of the node. Negative numbers indicate a
     * lean to the left, positive to the right, and 0 is perfectly
     * balanced.
     *
     * @return int
     */
    public function balance() : int
    {
        return ($this->right ? $this->right->height() : 0)
            - ($this->left ? $this->left->height() : 0);
    }

    /**
     * Set the parent of this node.
     *
     * @param self|null $node
     */
    public function setParent(?BinaryNode $node = null) : void
    {
        $this->parent = $node;
    }

    /**
     * Set the left child node.
     *
     * @param self $node
     */
    public function attachLeft(BinaryNode $node) : void
    {
        $node->setParent($this);

        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param self $node
     */
    public function attachRight(BinaryNode $node) : void
    {
        $node->setParent($this);

        $this->right = $node;
    }

    /**
     * Detach the left child node.
     */
    public function detachLeft() : void
    {
        if ($this->left) {
            $this->left->setParent(null);

            $this->left = null;
        }
    }

    /**
     * Detach the right child node.
     */
    public function detachRight() : void
    {
        if ($this->right) {
            $this->right->setParent(null);

            $this->right = null;
        }
    }

    /**
     * Is this an orphaned node?
     *
     * @return bool
     */
    public function orphan() : bool
    {
        return !$this->parent and $this->leaf();
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
