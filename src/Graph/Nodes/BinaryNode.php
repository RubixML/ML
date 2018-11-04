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
        return 1 + max(isset($this->left) ? $this->left->height() : 0,
            isset($this->right) ? $this->right->height() : 0);
    }

    /**
     * The balance factor of the node.
     *
     * @return int
     */
    public function balance() : int
    {
        return (isset($this->left) ? $this->left->height() : 0)
            - (isset($this->right) ? $this->right->height() : 0);
    }

    /**
     * Set the parent of this node.
     * 
     * @param  self|null  $node
     * @return void
     */
    public function setParent(?self $node = null) : void
    {
        $this->parent = $node;
    }

    /**
     * Set the left child node.
     *
     * @param  self  $node
     * @return void
     */
    public function attachLeft(self $node) : void
    {
        $node->setParent($this);

        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param  self  $node
     * @return void
     */
    public function attachRight(self $node) : void
    {
        $node->setParent($this);

        $this->right = $node;
    }

    /**
     * Detach the left child node.
     *
     * @return void
     */
    public function detachLeft() : void
    {
        if (isset($this->left)) {
            $this->left->setParent(null);

            $this->left = null;
        }
    }

    /**
     * Detach the right child node.
     *
     * @return void
     */
    public function detachRight() : void
    {
        if (isset($this->right)) {
            $this->right->setParent(null);

            $this->right = null;
        }
    }

    /**
     * Is this a leaf node? i.e no children.
     *
     * @return bool
     */
    public function leaf() : bool
    {
        return !isset($this->left) and !isset($this->right);
    }
}
