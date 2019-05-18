<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Embedders\Embedder;
use Rubix\ML\Other\Helpers\DataType;
use InvalidArgumentException;

class DatasetIsCompatibleWithEmbedder
{
    /**
     * Perform a check.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Embedders\Embedder $embedder
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Embedder $embedder) : void
    {
        $types = $dataset->uniqueTypes();
        $compatibility = $embedder->compatibility();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $different = array_diff($types, $compatibility);

            $diff = implode(', ', array_map(function ($type) {
                return DataType::TYPES[$type];
            }, $different));

            $compat = implode(', ', array_map(function ($type) {
                return DataType::TYPES[$type];
            }, $compatibility));

            throw new InvalidArgumentException('Embedder is not'
                . " compatible with $diff data type"
                . (count($different) > 1 ? 's.' : '.')
                . " Compatible data types are $compat.");
        }
    }
}
