<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;
use Stringable;

/**
 * Seeder
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Seeder extends Stringable
{
    /**
     * Seed k cluster centroids from a dataset.
     *
     * @internal
     *
     * @param Dataset $dataset
     * @param int $k
     * @return list<list<string|int|float>>
     */
    public function seed(Dataset $dataset, int $k) : array;
}
