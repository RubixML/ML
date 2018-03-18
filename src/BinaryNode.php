<?php

namespace Rubix\Engine;

use InvalidArgumentException;

class BinaryNode extends GraphObject implements Node
{
    /**
     * The parent node.
     *
     * @var \Rubix\Engine\BinaryNode|null
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
     * @param  mixed  $value
     * @param  array  $properties
     * @return void
     */
    public function __construct($value, array $properties = [])
    {
        $this->changeValue($value);
        
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
     * Change the value of the node.
     *
     * @param  mixed  $values
     * @return self
     */
    public function changeValue($value) : self
    {
        if (!is_numeric($value) && !is_string($value)) {
            throw new InvalidArgumentException('Value must be a string or numeric type, ' . gettype($value) . ' found.');
        }

        $this->value = $value;

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
        $this->detachLeft();

        if (isset($node)) {
            $node->setParent($this);

            $this->left = $node;
        }

        $this->updateHeight();

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
        $this->detachRight();

        if (isset($node)) {
            $node->setParent($this);

            $this->right = $node;
        }

        $this->updateHeight();

        return $this;
    }

    /**
     * Detach the left child node.
     *
     * @return self
     */
    public function detachLeft() : self
    {
        if (isset($this->left)) {
            $this->left->setParent(null);

            $this->left = null;

            $this->updateHeight();
        }

        return $this;
    }

    /**
     * Detach the right child node.
     *
     * @return self
     */
    public function detachRight() : self
    {
        if (isset($this->right)) {
            $this->right->setParent(null);

            $this->right = null;

            $this->updateHeight();
        }

        return $this;
    }

    /**
     * Recursive function to update the node's height up to the root. O(logN)
     *
     * @return self
     */
    public function updateHeight() : self
    {
        $height = 1 + max(isset($this->left) ? $this->left->height() : 0,
            isset($this->right) ? $this->right->height() : 0);

        if ($this->height !== $height) {
            $this->height = $height;

            if (!is_null($this->parent())) {
                $this->parent()->updateHeight();
            }
        }

        return $this;
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
