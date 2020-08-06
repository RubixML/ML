<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class DatasetHasDimensionality
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $dimensions
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, int $dimensions) : void
    {
        if ($dataset->numColumns() !== $dimensions) {
            throw new InvalidArgumentException(
                'Dataset must contain'
                . " samples with exactly $dimensions dimensions,"
                . " {$dataset->numColumns()} given."
            );
        }
    }
}
