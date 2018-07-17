<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

class BinaryNode implements Node
{
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
     * Set the left child node.
     *
     * @param  self  $node
     * @return void
     */
    public function attachLeft(self $node) : void
    {
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
        $this->right = $node;
    }

    /**
     * Detach the left child node.
     *
     * @return void
     */
    public function detachLeft() : void
    {
        $this->left = null;
    }

    /**
     * Detach the right child node.
     *
     * @return void
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
    public function isLeaf() : bool
    {
        return is_null($this->left) and is_null($this->right);
    }
}
