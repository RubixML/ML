<?php

namespace Rubix\ML\Extractors;

/**
 * Writer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Writer
{
    /**
     * Write an iterable data table to disk.
     *
     * @param iterable<mixed[]> $iterator
     * @param string[]|null $header
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function write(iterable $iterator, ?array $header = null) : void;
}
