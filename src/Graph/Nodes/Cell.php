<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Cell
 *
 * A cell node contains samples that are likely members of the same
 * group.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cell extends BinaryNode implements Leaf
{
    /**
     * The estimated depth of this node.
     *
     * @var float
     */
    protected $depth;

    /**
     * The number of training points located in this cell.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  float  $depth
     * @param  int  $n
     * @return void
     */
    public function __construct(float $depth, int $n)
    {
        $this->depth = $depth;
        $this->n = $n;
    }

    /**
     * Return the estimated depth of this node.
     *
     * @return float
     */
    public function depth() : float
    {
        return $this->depth;
    }

    /**
     * Return the number of training points located in this cell.
     *
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }
}
