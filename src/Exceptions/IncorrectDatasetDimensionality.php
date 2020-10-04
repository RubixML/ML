<?php

namespace Rubix\ML\Exceptions;

use Rubix\ML\Datasets\Dataset;

class IncorrectDatasetDimensionality extends InvalidArgumentException
{
    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $dimensions
     */
    public function __construct(Dataset $dataset, int $dimensions)
    {
        $message = 'Dataset must contain samples with'
            . " exactly $dimensions dimensions,"
            . " {$dataset->numColumns()} given.";

        parent::__construct($message);
    }
}
