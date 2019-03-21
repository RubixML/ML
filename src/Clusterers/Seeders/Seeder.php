<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;

interface Seeder
{
    public const EPSILON = 1e-8;
    
    /**
     * Seed k cluster centroids from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @return array
     */
    public function seed(Dataset $dataset, int $k) : array;
}
