<?php

namespace Rubix\Engine;

use InvalidArgumentException;

class BinaryNode extends GraphObject
{
    const BALANCE_FACTOR = 1;

    /**
     * The parent node.
     *
     * @var \Rubix\Engine\BinaryNode
     */
    protected $parent;

    /**
     * The value of the node.
     *
     * @var mixed
     */
    protected $value;

    /**
     * The left child node.
     *
     * @var \Rubix\Engine\BinaryNode
     */
    protected $left;

    /**
     * The right child node.
     *
     * @var \Rubix\Engine\BinaryNode
     */
    protected $right;

    /**
     * @param  mixed  $value
     * @param  array  $properties
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($value, array $properties = [])
    {
        if (!is_numeric($value) && !is_string($value)) {
            throw new InvalidArgumentException('Value must be a numeric or string type, ' . gettype($value) . ' found.');
        }

        $this->parent = null;
        $this->value = $value;
        $this->left = null;
        $this->right = null;

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
     * @return mixed
     */
    public function value()
    {
        return $this->value;
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
     * @param  \Rubix\Engine\BinaryNode|null  $parent
     * @return self
     */
    public function setParent(BinaryNode $parent = null) : BinaryNode
    {
        $this->parent = $parent;

        return $this;
    }

    /**
     * Set the left child node.
     *
     * @param  \Rubix\Engine\BinaryNode|null  $node
     * @return self
     */
    public function attachLeft(BinaryNode $node = null) : BinaryNode
    {
        $this->left = $node;

        return $this;
    }

    /**
     * Set the right child node.
     *
     * @param  \Rubix\Engine\BinaryNode|null  $node
     * @return self
     */
    public function attachRight(BinaryNode $node = null) : BinaryNode
    {
        $this->right = $node;

        return $this;
    }

    /**
     * Rotates the node to the left and returns the new root.
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return \Rubix\Engine\BinaryNode
     */
    public function rotateLeft() : BinaryNode
    {
        $y = $this->right;
        $temp = $y->left();

        $y->attachLeft($this);
        $this->attachRight($temp);

        return $y;
    }

    /**
     * Rotates the node to the right and returns the new root.
     *
     * @param  \Rubix\Engine\BinaryNode  $node
     * @return \Rubix\Engine\BinaryNode
     */
    public function rotateRight() : BinaryNode
    {
        $x = $this->left;
        $temp = $x->right();

        $x->attachRight($this);
        $this->attachLeft($temp);

        return $x;
    }

    /**
     * Is the node balanced?
     *
     * @return bool
     */
    public function isBalanced() : bool
    {
        return abs($this->balance()) <= static::BALANCE_FACTOR;
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
