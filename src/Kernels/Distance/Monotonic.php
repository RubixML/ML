<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Monotonic
 *
 * A marker interface for distance kernels that are coordinatewise monotone non-decreasing
 * in the absolute difference between sample features. Such kernels are safe to use with
 * spatial trees that prune branches using axis-aligned bounding boxes such as the k-d tree
 * because clamping a sample into the bounds of a hypercube can only decrease the distance,
 * producing a valid lower bound.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Monotonic extends Distance
{
    //
}
