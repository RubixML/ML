<?php

namespace Rubix\ML\Datasets\Extractors;

interface Extractor
{
    /**
     * Read the records and return them in an iterator.
     *
     * @return iterable<array>
     */
    public function extract() : iterable;

    /**
     * Set the row offset of the cursor.
     *
     * @param int $offset
     * @return self
     */
    public function setOffset(int $offset);

    /**
     * Set the maximum number of rows to return.
     *
     * @param int $limit
     * @return self
     */
    public function setLimit(int $limit);
}
