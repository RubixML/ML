<?php

namespace Rubix\ML\Graph;

class BinaryTree implements Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Node|null
     */
    protected $root;

    /**
     * @param  \Rubix\ML\Graph\BinaryNode  $node
     * @return void
     */
    public function __construct(BinaryNode $node = null)
    {
        $this->setRoot($node);
    }

    /**
     * @return \Rubix\ML\Graph\Node|null
     */
    public function root() : ?Node
    {
        return $this->root;
    }

    /**
     * Set the root node of the tree.
     *
     * @param  \Rubix\ML\Graph\BinaryNode|null  $node
     * @return void
     */
    public function setRoot(BinaryNode $node = null) : void
    {
        $this->root = $node;
    }

    /**
     * The height of the tree. O(V) because node heights are not memoized.
     *
     * @return int
     */
    public function height() : int
    {
        return isset($this->root) ? $this->root->height() : 0;
    }

    /**
     * The balance factor of the tree. O(V) because balance requires height of
     * each node.
     *
     * @return int
     */
    public function balance() : int
    {
        return isset($this->root) ? $this->root->balance() : 0;
    }

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool
    {
        return !isset($this->root);
    }
}
