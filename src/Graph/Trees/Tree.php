<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;

interface Tree
{
    /**
     * Insert a root node and recursively split the dataset until a
     * terminating condition is met.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     */
    public function grow(Dataset $dataset) : void;

    /**
     * Return the height of the tree i.e. the number of levels.
     *
     * @return int
     */
    public function height() : int;

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool;

    /**
     * Remove the root node and its descendants from the tree.
     */
    public function destroy() : void;
}
