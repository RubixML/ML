<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;

interface Tree
{
    /**
     * Return the root node of the tree.
     *
     * @return mixed
     */
    public function root();
    
    /**
     * Insert a root node and recursively split the dataset a terminating
     * condition is met.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function grow(Dataset $dataset) : void;

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool;

    /**
     * Destroy the tree.
     */
    public function destroy() : void;
}
