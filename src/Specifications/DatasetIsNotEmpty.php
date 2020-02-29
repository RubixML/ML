<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class DatasetIsNotEmpty
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset) : void
    {
        if ($dataset->empty()) {
            throw new InvalidArgumentException('Dataset must contain'
                . ' at least one record.');
        }
    }
}
