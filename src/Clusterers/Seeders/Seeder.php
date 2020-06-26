<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;

interface Seeder
{
    /**
     * Seed k cluster centroids from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @return array[]
     */
    public function seed(Dataset $dataset, int $k) : array;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
