<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

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
     * @param  float  $depth
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $depth)
    {
        if ($depth < 0.) {
            throw new InvalidArgumentException("Depth cannot be less"
                . " than 0, $depth given.");
        }

        $this->depth = $depth;
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
}
