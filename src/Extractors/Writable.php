<?php

namespace Rubix\ML\Extractors;

/**
 * Writable
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Writable
{
    /**
     * Write an iterable data table to disk.
     *
     * @param iterable<mixed[]> $iterator
     */
    public function write(iterable $iterator) : void;
}
