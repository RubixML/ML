<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

use function count;

class SamplesAreCompatibleWithEstimator
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Estimator $estimator
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Estimator $estimator) : void
    {
        $compatibility = $estimator->compatibility();

        $types = $dataset->uniqueTypes();

        $compatible = array_intersect($types, $compatibility);

        if (count($compatible) < count($types)) {
            $incompatible = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                "$estimator is not compatible with " . implode(', ', $incompatible) . ' data types.'
            );
        }
    }
}
