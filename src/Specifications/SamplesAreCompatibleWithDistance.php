<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

use function count;

class SamplesAreCompatibleWithDistance
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Distance $kernel) : void
    {
        $compatibility = $kernel->compatibility();

        $types = $dataset->uniqueTypes();

        $compatible = array_intersect($types, $compatibility);

        if (count($compatible) < count($types)) {
            $incompatible = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                "$kernel is not compatible with " . implode(', ', $incompatible) . ' data types.'
            );
        }
    }
}
