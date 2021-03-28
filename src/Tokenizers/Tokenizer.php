<?php

namespace Rubix\ML\Tokenizers;

use Stringable;

/**
 * Tokenizer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Tokenizer extends Stringable
{
    /**
     * Tokenize a blob of text.
     *
     * @internal
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array;
}
