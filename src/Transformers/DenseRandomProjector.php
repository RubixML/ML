<?php

namespace Rubix\ML\Transformers;

/**
 * Dense Random Projector
 *
 * The Dense Random Projector uses a random matrix sampled from a dense uniform
 * distribution [-1, 1] to project a sample matrix onto a target dimensionality.
 *
 * References:
 * [1] D. Achlioptas. (2003). Database-friendly random projections:
 * Johnson-Lindenstrauss with binary coins.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DenseRandomProjector extends SparseRandomProjector
{
    /**
     * The numbers to draw from when generating the random matrix.
     *
     * @var float[]
     */
    protected const DISTRIBUTION = [-1.0, 1.0];
}
