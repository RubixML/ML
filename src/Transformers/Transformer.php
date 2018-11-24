<?php

namespace Rubix\ML\Transformers;

interface Transformer
{
    const EPSILON = 1e-8;

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void;
}
