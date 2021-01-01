<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * @internal
 */
class SamplesAreCompatibleWithEstimator extends Specification
{
    /**
     * The dataset that contains samples under validation.
     *
     * @var \Rubix\ML\Datasets\Dataset
     */
    protected $dataset;

    /**
     * The estimator.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $estimator;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Estimator $estimator
     * @return self
     */
    public static function with(Dataset $dataset, Estimator $estimator) : self
    {
        return new self($dataset, $estimator);
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Estimator $estimator
     */
    public function __construct(Dataset $dataset, Estimator $estimator)
    {
        $this->dataset = $dataset;
        $this->estimator = $estimator;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        $compatibility = $this->estimator->compatibility();

        $types = $this->dataset->uniqueTypes();

        $compatible = array_intersect($types, $compatibility);

        if (count($compatible) < count($types)) {
            $incompatible = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                "{$this->estimator} is incompatible with " . implode(', ', $incompatible) . ' data types.'
            );
        }
    }
}
