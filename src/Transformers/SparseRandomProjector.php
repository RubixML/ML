<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use InvalidArgumentException;

/**
 * Sparse Random Projector
 *
 * The Sparse Random Projector uses a random matrix sampled from a sparse uniform
 * distribution (mostly 0s) to project a sample matrix onto a target dimensionality.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SparseRandomProjector extends GaussianRandomProjector
{
    const ROOT_3 = 1.73205080757;
    
    const DISTRIBUTION = [-self::ROOT_3, 0.0, 0.0, 0.0, 0.0, self::ROOT_3];

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $columns = $dataset->numColumns();

        $p = count(static::DISTRIBUTION) - 1;

        $r = [];

        for ($i = 0; $i < $columns; $i++) {
            for ($j = 0; $j < $this->dimensions; $j++) {
                $r[$i][] = static::DISTRIBUTION[rand(0, $p)];
            }
        }

        $this->r = Matrix::quick($r);
    }
}
