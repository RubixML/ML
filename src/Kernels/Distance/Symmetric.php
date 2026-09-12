<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Symmetric
 *
 * A marker interface for distance kernels that are symmetric, meaning the
 * distance between sample a and b equals the distance between b and a. Such
 * kernels can exploit symmetry when computing the full pairwise distance matrix
 * to halve the number of distance evaluations required.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Symmetric extends Distance
{
    //
}
