<?php

namespace Rubix\Engine;

use InvalidArgumentException;

class BinaryNode extends GraphObject
{
    const MAX_BALANCE_FACTOR = 1;

    /**
     * The parent node.
     *
     * @var \Rubix\Engine\BinaryNode|null
     */
    protected $parent;

    /**
     * The left child node.
     *
     * @var \Rubix\Engine\BinaryNode|null
     */
    protected $left;

    /**
     * The right child node.
     *
     * @var \Rubix\Engine\BinaryNode|null
     */
    protected $right;

    /**
     * The precomputed height of the node.
     *
     * @var int
     */
    protected $height;

    /**
     * @param  array  $properties
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $properties = [])
    {
        $this->parent = null;
        $this->left = null;
        $this->right = null;
        $this->height = 1;

        parent::__construct($properties);
    }

    /**
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function parent() : ?BinaryNode
    {
        return $this->parent;
    }

    /**
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function left() : ?BinaryNode
    {
        return $this->left;
    }

    /**
     * @return \Rubix\Engine\BinaryNode|null
     */
    public function right() : ?BinaryNode
    {
        return $this->right;
    }

    /**
     * @return int
     */
    public function height() : int
    {
        return $this->height;
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
     * @param  \Rubix\Engine\BinaryNode|null  $node
     * @return self
     */
    public function setParent(BinaryNode $node = null) : self
    {
        $this->parent = $node;

        return $this;
    }

    /**
     * Set the left child node.
     *
     * @param  \Rubix\Engine\BinaryNode|null  $node
     * @return self
     */
    public function attachLeft(BinaryNode $node = null) : self
    {
        $this->left = $node;

        $this->updateHeight();

        if (isset($node)) {
            $node->setParent($this);
        }

        return $this;
    }

    /**
     * Set the right child node.
     *
     * @param  \Rubix\Engine\BinaryNode|null  $node
     * @return self
     */
    public function attachRight(BinaryNode $node = null) : self
    {
        $this->right = $node;

        $this->updateHeight();

        if (isset($node)) {
            $node->setParent($this);
        }

        return $this;
    }

    /**
     * Detach the left child node.
     *
     * @return self
     */
    public function detachLeft() : self
    {
        if (!isset($this->left)) {
            return $this;
        }

        $this->left->setParent(null);

        $this->left = null;

        $this->updateHeight();

        return $this;
    }

    /**
     * Detach the right child node.
     *
     * @return self
     */
    public function detachRight() : self
    {
        if (!isset($this->right)) {
            return $this;
        }

        $this->right->setParent(null);

        $this->right = null;

        $this->updateHeight();

        return $this;
    }

    /**
     * Update the height of the node.
     *
     * @return void
     */
    public function updateHeight() : self
    {
        $this->height = 1 + max(isset($this->left) ? $this->left->height() : 0,
            isset($this->right) ? $this->right->height() : 0);

        return $this;
    }

    /**
     * Is the node balanced?
     *
     * @return bool
     */
    public function isBalanced() : bool
    {
        return abs($this->balance()) <= static::MAX_BALANCE_FACTOR;
    }

    /**
     * Is this a leaf node? I.e no children.
     *
     * @return bool
     */
    public function isLeaf() : bool
    {
        return !isset($this->left) && !isset($this->right);
    }
}
