<?php

namespace Rubix\ML\Transformers;

use Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;

use function count;

/**
 * Sparse Random Projector
 *
 * The Sparse Random Projector uses a random matrix sampled from a sparse uniform
 * distribution (mostly 0s) to project a sample matrix onto a target dimensionality.
 *
 * References:
 * [1] D. Achlioptas. (2003). Database-friendly random projections:
 * Johnson-Lindenstrauss with binary coins.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SparseRandomProjector extends GaussianRandomProjector
{
    /**
     * The square root of 3.
     *
     * @var float
     */
    protected const ROOT_3 = 1.73205080757;
    
    /**
     * The numbers to draw from when generating the random matrix.
     *
     * @var float[]
     */
    protected const DISTRIBUTION = [-self::ROOT_3, 0.0, 0.0, 0.0, 0.0, self::ROOT_3];

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        $columns = $dataset->numColumns();

        $p = count(static::DISTRIBUTION) - 1;

        $r = [];

        while (count($r) < $columns) {
            $row = [];

            while (count($row) < $this->dimensions) {
                $row[] = static::DISTRIBUTION[rand(0, $p)];
            }

            $r[] = $row;
        }

        $this->r = Matrix::quick($r);
    }
}
