<?php

namespace Rubix\Engine;

class Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\Engine\Node|null  $root
     */
    protected $root;

    /**
     * @param \Rubix\Engine\Node|null
     * @return void
     */
    public function __construct(Node $root = null)
    {
        $this->root = $root;
    }

    /**
     * @return \Rubix\Engine\Node|null
     */
    public function root() : ?Node
    {
        return $this->root;
    }

    /**
     * Is the tree empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return is_null($this->root);
    }
}
