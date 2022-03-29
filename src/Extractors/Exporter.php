<?php

namespace Rubix\ML\Extractors;

/**
 * Exporter
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Exporter
{
    /**
     * Export an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     */
    public function export(iterable $iterator) : void;
}
