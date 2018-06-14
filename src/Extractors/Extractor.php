<?php

namespace Rubix\ML\Extractors;

interface Extractor
{
    /**
     * Fit the extractor to the raw sample data.
     *
     * @param  array  $samples
     * @return void
     */
    public function fit(array $samples) : void;

    /**
     * Extract features from raw samples and return an array of vectors.
     *
     * @param  array  $samples
     * @return array
     */
    public function extract(array $samples) : array;
}
