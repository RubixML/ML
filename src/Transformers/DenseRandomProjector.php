<?php

namespace Rubix\ML\Transformers;

use function Rubix\ML\warn_deprecated;

/**
 * Dense Random Projector
 *
 * The Dense Random Projector uses a random matrix sampled from a dense uniform
 * distribution [-1, 1] to project a sample matrix onto a target dimensionality.
 *
 * References:
 * [1] D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss
 * with binary coins.
 *
 * @deprecated
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DenseRandomProjector extends SparseRandomProjector
{
    /**
     * @param int $dimensions
     */
    public function __construct(int $dimensions)
    {
        warn_deprecated('Dense Random Projector is deprecated, use Sparse Random Projector with sparsity set to 0 instead.');

        parent::__construct($dimensions, 0.0);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Dense Random Projector (dimensions: {$this->dimensions})";
    }
}
