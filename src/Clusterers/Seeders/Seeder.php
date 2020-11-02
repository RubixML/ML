<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;

/**
 * Seeder
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Seeder
{
    /**
     * Seed k cluster centroids from a dataset.
     *
     * @internal
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @return list<list<string|int|float>>
     */
    public function seed(Dataset $dataset, int $k) : array;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
