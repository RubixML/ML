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
     * Export an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     */
    public function export(iterable $iterator) : void;
}
