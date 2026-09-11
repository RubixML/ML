<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Subadditive
 *
 * A marker interface for distance kernels that satisfy the triangle
 * inequality otherwise known as a *metric*. Such kernels are safe to use with
 * spatial trees that rely on the triangle inequality to bound their search
 * space such as the ball tree and vantage tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Subadditive extends Distance
{
    //
}
