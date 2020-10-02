<?php

namespace Rubix\ML\Transformers;

use Stringable;

interface Transformer extends Stringable
{
    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array;

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     */
    public function transform(array &$samples) : void;
}
