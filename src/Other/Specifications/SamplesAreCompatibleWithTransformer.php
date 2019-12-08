<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Transformers\Transformer;
use InvalidArgumentException;

use function count;

class SamplesAreCompatibleWithTransformer
{
    /**
     * Perform a check of the specification.
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
            $diff = array_diff($types, $compatibility);

            $diffString = implode(', ', array_map([DataType::class, 'asString'], $diff));

            throw new InvalidArgumentException(Params::shortName($transformer)
                . " is not compatible with $diffString data types.");
        }
    }
}
