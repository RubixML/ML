<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

class DatasetIsCompatibleWithDistance
{
    /**
     * Perform a check.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Distance $kernel) : void
    {
        $types = $dataset->uniqueTypes();

        $compatibility = $kernel->compatibility();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $different = array_diff($types, $compatibility);

            $diffString = implode(', ', array_map([DataType::class, 'asString'], $different));

            $compatString = implode(', ', array_map([DataType::class, 'asString'], $compatibility));

            throw new InvalidArgumentException('Distance kernel is not'
                . " compatible with $diffString data type"
                . (count($different) > 1 ? 's.' : '.')
                . " Compatible data types are $compatString.");
        }
    }
}
