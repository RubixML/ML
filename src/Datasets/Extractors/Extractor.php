<?php

namespace Rubix\ML\Datasets\Extractors;

interface Extractor
{
    /**
     * Extract and build a dataset object from source.
     *
     * @param int $offset
     * @param int $limit
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function extract(int $offset = 0, ?int $limit = null);
}
