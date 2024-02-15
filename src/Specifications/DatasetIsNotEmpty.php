<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Exceptions\EmptyDataset;

/**
 * @internal
 */
class DatasetIsNotEmpty extends Specification
{
    /**
     * The dataset.
     *
     * @var Dataset
     */
    protected Dataset $dataset;

    /**
     * Build a specification object with the given arguments.
     *
     * @param Dataset $dataset
     * @return self
     */
    public static function with(Dataset $dataset) : self
    {
        return new self($dataset);
    }

    /**
     * @param Dataset $dataset
     */
    public function __construct(Dataset $dataset)
    {
        $this->dataset = $dataset;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws EmptyDataset
     */
    public function check() : void
    {
        if ($this->dataset->empty()) {
            throw new EmptyDataset();
        }
    }
}
