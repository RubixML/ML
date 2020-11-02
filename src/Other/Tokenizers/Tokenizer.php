<?php

namespace Rubix\ML\Other\Tokenizers;

/**
 * Tokenizer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Tokenizer
{
    /**
     * Tokenize a blob of text.
     *
     * @internal
     *
     * @param string $string
     * @return list<string>
     */
    public function tokenize(string $string) : array;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
