<?php

namespace Rubix\ML\Graph\Trees;

interface Tree
{
    /**
     * Return the root node of the tree.
     *
     * @return mixed
     */
    public function root();

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool;
}
