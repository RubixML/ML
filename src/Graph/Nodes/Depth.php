<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;

/**
 * Depth
 *
 * A node that estimates the depth that a sample reaches in the tree when performing an
 * unsuccessful search. The depth is estimated using a combination of the actual depth at the
 * point of termination plus an approximation based on the number of samples that are left to
 * isolate.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Depth implements BinaryNode
{
    /**
     * The approximated depth of this node.
     *
     * @var float
     */
    protected float $depth;

    /**
     * Estimate the average path length of an unsuccessful search given n unisolated samples.
     *
     * @param int $n
     * @return float
     */
    public static function c(int $n) : float
    {
        switch (true) {
            case $n > 2:
                return 2.0 * (log($n - 1) + M_EULER) - 2.0 * ($n - 1) / $n;

            case $n === 2:
                return 1.0;

            default:
                return 0.0;
        }
    }

    /**
     * Terminate a branch with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $depth
     * @return self
     */
    public static function terminate(Dataset $dataset, int $depth) : self
    {
        return new self($depth + self::c($dataset->numSamples()) - 1.0);
    }

    /**
     * @param float $depth
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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

    /**
     * Return the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return 1;
    }
}
