<?php

namespace Rubix\ML\Tokenizers\Stemmers;

interface Stemmer
{
    /**
     * Stem a word to its root form.
     *
     * @param string $word
     * @return string
     */
    public function stem(string $word) : string;
}
