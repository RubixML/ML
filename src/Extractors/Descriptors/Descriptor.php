<?php

namespace Rubix\ML\Extractors\Descriptors;

interface Descriptor
{
    /**
     * Extract features from the image and return them in a vector.
     *
     * @param  array  $volume
     * @return array
     */
    public function describe(array $volume) : array;
}
