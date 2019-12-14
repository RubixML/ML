<?php

namespace Rubix\ML\Datasets\Extractors;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;

interface Extractor
{
    /**
     * Extract and build an unlabeled dataset object from source.
     *
     * @param int $offset
     * @param int|null $limit
     * @return \Rubix\ML\Datasets\Unlabeled
     */
    public function extract(int $offset = 0, ?int $limit = null) : Unlabeled;

    /**
     * Extract and build a labeled dataset object from source.
     *
     * @param int $offset
     * @param int|null $limit
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function extractWithLabels(int $offset = 0, ?int $limit = null) : Labeled;
}
