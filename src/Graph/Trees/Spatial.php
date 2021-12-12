<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;
use Stringable;

/**
 * Spatial
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Spatial extends Tree, Stringable
{
    /**
     * Return the distance kernel used to compute distances.
     *
     * @internal
     *
     * @return \Rubix\ML\Kernels\Distance\Distance
     */
    public function kernel() : Distance;

    /**
     * Insert a root node and recursively split the dataset until a terminating condition is met.
     *
     * @internal
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function grow(Labeled $dataset) : void;

    /**
     * Run a k nearest neighbors search and return the samples, labels, and distances in a 3-tuple.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @param int $k
     * @return array{list<list<mixed>>,list<mixed>,list<float>}
     */
    public function nearest(array $sample, int $k) : array;

    /**
     * Return all samples, labels, and distances within a given radius of a sample.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @param float $radius
     * @return array{list<list<mixed>>,list<mixed>,list<float>}
     */
    public function range(array $sample, float $radius) : array;

    /**
     * Remove the root node and its descendants from the tree.
     *
     * @internal
     */
    public function destroy() : void;
}
