<?php

namespace Rubix\Engine;

class BinaryNode extends GraphObject
{
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
     * Recursive function to update the node's height up to the root. O(N logN)
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
