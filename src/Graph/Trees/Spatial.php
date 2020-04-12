<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;

interface Spatial extends Tree
{
    /**
     * Return the distance kernel used to compute distances.
     *
     * @return \Rubix\ML\Kernels\Distance\Distance
     */
    public function kernel() : Distance;

    /**
     * Insert a root node and recursively split the dataset until a
     * terminating condition is met.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function grow(Labeled $dataset) : void;

    /**
     * Run a k nearest neighbors search and return the samples, labels, and
     * distances in a 3-tuple.
     *
     * @param (string|int|float)[] $sample
     * @param int $k
     * @return array[]
     */
    public function nearest(array $sample, int $k) : array;

    /**
     * Return all samples, labels, and distances within a given radius of a
     * sample.
     *
     * @param (string|int|float)[] $sample
     * @param float $radius
     * @return array[]
     */
    public function range(array $sample, float $radius) : array;

    /**
     * Remove the root node and its descendants from the tree.
     */
    public function destroy() : void;
}
