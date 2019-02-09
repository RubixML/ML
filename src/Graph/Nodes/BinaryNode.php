<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

/**
 * Binary Node
 *
 * A node of a binary tree i.e a tree whose parents have a maximum of
 * two immediate children per node.
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
     * @var self|null
     */
    protected $parent;

    /**
     * The left child node.
     *
     * @var self|null
     */
    protected $left;

    /**
     * The right child node.
     *
     * @var self|null
     */
    protected $right;

    /**
     * Return the parent node.
     *
     * @return self|null
     */
    public function parent() : ?self
    {
        return $this->parent;
    }

    /**
     * Return the children of this node in an array.
     *
     * @return array
     */
    public function children() : array
    {
        $children = [];

        if ($this->left) {
            $children[] = $this->left;
        }
        if ($this->right) {
            $children[] = $this->right;
        }

        return $children;
    }

    /**
     * Return the left child node.
     *
     * @return self|null
     */
    public function left() : ?self
    {
        return $this->left;
    }

    /**
     * Return the right child node.
     *
     *
     * @return self|null
     */
    public function right() : ?self
    {
        return $this->right;
    }

    /**
     * Recursive function to determine the height of the node.
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
     * The balance factor of the node.
     *
     * @return int
     */
    public function balance() : int
    {
        return ($this->left ? $this->left->height() : 0)
            - ($this->right ? $this->right->height() : 0);
    }

    /**
     * Set the parent of this node.
     *
     * @param self|null $node
     */
    public function setParent(?self $node = null) : void
    {
        $this->parent = $node;
    }

    /**
     * Set the left child node.
     *
     * @param self $node
     */
    public function attachLeft(self $node) : void
    {
        $node->setParent($this);

        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param self $node
     */
    public function attachRight(self $node) : void
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
     * Is this node an orphaned node?
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
