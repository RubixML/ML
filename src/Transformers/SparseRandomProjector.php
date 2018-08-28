<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Structures\Matrix;
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
    const DISTRIBUTION = [-self::ROOT_3, 0.0, 0.0, 0.0, 0.0, self::ROOT_3];

    const ROOT_3 = 1.73205080757;

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(Dataset::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $columns = $dataset->numColumns();

        $n = count(static::DISTRIBUTION) - 1;

        $r = [[]];

        for ($i = 0; $i < $columns; $i++) {
            for ($j = 0; $j < $this->dimensions; $j++) {
                $r[$i][$j] = static::DISTRIBUTION[rand(0, $n)];
            }
        }

        $this->r = new Matrix($r);
    }
}
