<?php

namespace Rubix\ML\Other\Tokenizers;

interface Tokenizer
{
    /**
     * Tokenize a blob of text.
     *
     * @param string $string
     * @return string[]
     */
    public function tokenize(string $string) : array;
}
