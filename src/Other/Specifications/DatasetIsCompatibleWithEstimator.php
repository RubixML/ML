<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use InvalidArgumentException;

class DatasetIsCompatibleWithEstimator
{
    /**
     * Perform a check.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Estimator $estimator
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Estimator $estimator) : void
    {
        $compatibility = $estimator->compatibility();

        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $different = array_diff($types, $compatibility);

            $diffString = implode(', ', array_map([DataType::class, 'asString'], $different));

            $compatString = implode(', ', array_map([DataType::class, 'asString'], $compatibility));

            throw new InvalidArgumentException('Estimator is not'
                . " compatible with $diffString data type"
                . (count($different) > 1 ? 's.' : '.')
                . " Compatible data types are $compatString.");
        }
    }
}
