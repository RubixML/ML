<?php

namespace Rubix\Engine\Transformers\Tokenizers;

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
        $tokens = [];

        preg_match_all('/\w\w+/u', $string, $tokens);

        return $tokens[0];
    }
}
