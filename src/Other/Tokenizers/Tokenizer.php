<?php

namespace Rubix\ML\Other\Tokenizers;

interface Tokenizer
{
    /**
     * Tokenize a block of text.
     *
     * @param  string  $string
     * @return array
     */
    public function tokenize(string $string) : array;
}
