<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Transformers\Transformer;
use InvalidArgumentException;

use function count;
use function get_class;

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
        $compatibility = $transformer->compatibility();
        
        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                Params::shortName(get_class($transformer))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
