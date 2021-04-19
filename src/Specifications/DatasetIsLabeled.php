<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\LabelsAreMissing;

/**
 * @internal
 */
class DatasetIsLabeled extends Specification
{
    /**
     * The dataset under validation.
     *
     * @var \Rubix\ML\Datasets\Dataset
     */
    protected \Rubix\ML\Datasets\Dataset $dataset;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public static function with(Dataset $dataset) : self
    {
        return new self($dataset);
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Dataset $dataset)
    {
        $this->dataset = $dataset;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\LabelsAreMissing
     */
    public function check() : void
    {
        if (!$this->dataset instanceof Labeled) {
            throw new LabelsAreMissing();
        }
    }
}
