<?php

namespace Rubix\ML\Graph\Trees;

/**
 * Tree
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Tree
{
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
}
