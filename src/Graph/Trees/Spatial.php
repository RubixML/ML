<?php

namespace Rubix\ML\Graph\Trees;

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
     * Run a k nearest neighbors search and return the samples, labels, and
     * distances in a 3-tuple.
     *
     * @param array $sample
     * @param int $k
     * @return array[]
     */
    public function nearest(array $sample, int $k) : array;

    /**
     * Return all samples, labels, and distances within a given radius of a
     * sample.
     *
     * @param array $sample
     * @param float $radius
     * @return array[]
     */
    public function range(array $sample, float $radius) : array;
}
