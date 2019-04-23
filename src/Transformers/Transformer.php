<?php

namespace Rubix\ML\Transformers;

interface Transformer
{
    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     */
    public function transform(array &$samples) : void;
}
