<?php

namespace Rubix\Engine\Tokenizers;

class WordTokenizer implements Tokenizer
{
    /**
     * Tokenize a string.
     *
     * @param  string  $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        return preg_split('/\w\w+/u', $string);
    }
}
