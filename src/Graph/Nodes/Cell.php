<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * Cell
 *
 * A cell node contains samples that are likely members of the same group.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cell extends BinaryNode implements Leaf
{
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
     * @return self
     */
    public static function terminate(Dataset $dataset, int $depth) : self
    {
        $depth += self::c($dataset->numRows()) - 1.;

        return new self($depth);
    }

    /**
     * Calculate the average path length of an unsuccessful search for n nodes.
     *
     * @param int $n
     * @return float
     */
    protected static function c(int $n) : float
    {
        if ($n <= 1) {
            return 1.;
        }
        
        return 2. * (log($n - 1) + M_EULER) - 2. * ($n - 1) / $n;
    }

    /**
     * @param float $depth
     * @throws \InvalidArgumentException
     */
    public function __construct(float $depth)
    {
        if ($depth < 0.) {
            throw new InvalidArgumentException('Depth cannot be less'
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
