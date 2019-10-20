<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Transformers\Transformer;
use InvalidArgumentException;

class DatasetIsCompatibleWithTransformer
{
    /**
     * Perform a check.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Transformers\Transformer $transformer
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Transformer $transformer) : void
    {
        $types = $dataset->uniqueTypes();

        $compatibility = $transformer->compatibility();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $different = array_diff($types, $compatibility);

            $diffString = implode(', ', array_map([DataType::class, 'asString'], $different));

            $compatString = implode(', ', array_map([DataType::class, 'asString'], $compatibility));

            throw new InvalidArgumentException('Transformer is not'
                . " compatible with $diffString data type"
                . (count($different) > 1 ? 's.' : '.')
                . " Compatible data types are $compatString.");
        }
    }
}
