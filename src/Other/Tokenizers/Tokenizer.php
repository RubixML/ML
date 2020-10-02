<?php

namespace Rubix\ML\Other\Tokenizers;

use Stringable;

interface Tokenizer extends Stringable
{
    /**
     * Tokenize a blob of text.
     *
     * @param string $text
     * @return list<string>
     */
    public function tokenize(string $text) : array;
}
