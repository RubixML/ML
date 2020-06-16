<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;

/**
 * Cell
 *
 * A cell node contains samples that are likely members of the same group.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cell implements BinaryNode, Leaf
{
    use HasBinaryChildren;

    /**
     * The approximated depth of this node.
     *
     * @var float
     */
    protected $depth;

    /**
     * Terminate a branch with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $depth
     * @return self
     */
    public static function terminate(Dataset $dataset, int $depth) : self
    {
        return new self($depth + ITree::c($dataset->numRows()) - 1.0);
    }

    /**
     * @param float $depth
     * @throws \InvalidArgumentException
     */
    public function __construct(float $depth)
    {
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
