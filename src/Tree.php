<?php

namespace Rubix\Graph;

class Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\Graph\Node|null  $root
     */
    protected $root;

    /**
     * @return void
     */
    public function __construct(Node $root = null)
    {
        $this->root = $root;
    }

    /**
     * @return \Rubix\Graph\NodeNode|null
     */
    public function root() : ?Node
    {
        return $this->root;
    }

    /**
     * Is the trie empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return !isset($this->root);
    }
}
