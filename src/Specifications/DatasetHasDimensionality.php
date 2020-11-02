<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * @internal
 */
class DatasetHasDimensionality extends Specification
{
    /**
     * The dataset that contains samples under validation.
     *
     * @var \Rubix\ML\Datasets\Dataset
     */
    protected $dataset;

    /**
     * The target dimensionality.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $dimensions
     * @return self
     */
    public static function with(Dataset $dataset, int $dimensions) : self
    {
        return new self($dataset, $dimensions);
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $dimensions
     * @throws \InvalidArgumentException
     */
    public function __construct(Dataset $dataset, int $dimensions)
    {
        if ($dimensions < 0) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        $this->dataset = $dataset;
        $this->dimensions = $dimensions;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \InvalidArgumentException
     */
    public function check() : void
    {
        if ($this->dataset->numColumns() !== $this->dimensions) {
            throw new InvalidArgumentException(
                'Dataset must contain'
                . " samples with exactly {$this->dimensions}"
                . " dimensions, {$this->dataset->numColumns()} given."
            );
        }
    }
}
