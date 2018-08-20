<?php

namespace Rubix\ML\Extractors\Descriptors;

interface Descriptor
{
    /**
     * Extract features from an image patch and return them in an array.
     *
     * @param  array  $patch
     * @return array
     */
    public function describe(array $patch) : array;
}
