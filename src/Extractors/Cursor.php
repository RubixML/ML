<?php

namespace Rubix\ML\Extractors;

abstract class Cursor extends Extractor
{
    /**
     * The row offset of the cursor.
     *
     * @var int
     */
    protected $offset = 0;

    /**
     * The maximum number of rows to return.
     *
     * @var int
     */
    protected $limit = PHP_INT_MAX;

    /**
     * Set the row offset of the cursor.
     *
     * @param int $offset
     * @return self
     */
    public function setOffset(int $offset) : self
    {
        $this->offset = $offset;

        return $this;
    }

    /**
     * Set the maximum number of rows to return.
     *
     * @param int $limit
     * @return self
     */
    public function setLimit(int $limit) : self
    {
        $this->limit = $limit;

        return $this;
    }
}
