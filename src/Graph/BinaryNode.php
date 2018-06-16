<?php

namespace Rubix\ML\Graph;

use InvalidArgumentException;

class BinaryNode extends GraphObject implements Node
{
    /**
     * The parent node.
     *
     * @var \Rubix\ML\Graph\BinaryNode|null
     */
    protected $parent;

    /**
     * The left child node.
     *
     * @var \Rubix\ML\Graph\BinaryNode|null
     */
    protected $left;

    /**
     * The right child node.
     *
     * @var \Rubix\ML\Graph\BinaryNode|null
     */
    protected $right;

    /**
     * @return \Rubix\ML\Graph\BinaryNode|null
     */
    public function parent() : ?self
    {
        return $this->parent;
    }

    /**
     * @return \Rubix\ML\Graph\BinaryNode|null
     */
    public function left() : ?self
    {
        return $this->left;
    }

    /**
     * @return \Rubix\ML\Graph\BinaryNode|null
     */
    public function right() : ?self
    {
        return $this->right;
    }

    /**
     * Recursive function to determine the height of the node. O(V)
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
     * Set the parent node.
     *
     * @param  \Rubix\ML\Graph\BinaryNode|null  $node
     * @return void
     */
    public function setParent(BinaryNode $node = null) : void
    {
        $this->parent = $node;
    }

    /**
     * Set the left child node.
     *
     * @param  \Rubix\ML\Graph\BinaryNode|null  $node
     * @return void
     */
    public function attachLeft(BinaryNode $node) : void
    {
        $this->detachLeft();

        $node->setParent($this);

        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param  \Rubix\ML\Graph\BinaryNode|null  $node
     * @return void
     */
    public function attachRight(BinaryNode $node) : void
    {
        $this->detachRight();

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
    public function isLeaf() : bool
    {
        return !isset($this->left) and !isset($this->right);
    }
}
