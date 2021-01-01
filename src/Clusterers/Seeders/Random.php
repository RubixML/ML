<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;

/**
 * Random
 *
 * Completely random selection of seeds from a given dataset.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Random implements Seeder
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
    public function seed(Dataset $dataset, int $k) : array
    {
        return $dataset->randomSubset($k)->samples();
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Random';
    }
}
