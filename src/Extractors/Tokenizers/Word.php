<?php

namespace Rubix\ML\Extractors\Tokenizers;

class Word implements Tokenizer
{
    /**
     * Should the text be normalized before tokenized? i.e. remove extra
     * whitespace and lowercase.
     *
     * @var bool
     */
    protected $normalize;

    /**
     * @param  bool  $normalize
     * @return void
     */
    public function __construct(bool $normalize = true)
    {
        $this->normalize = $normalize;
    }

    /**
     * Tokenize a string.
     *
     * @param  string  $string
     * @return array
     */
    public function tokenize(string $string) : array
    {
        if ($this->normalize) {
            $string = preg_replace('/\s+/', ' ', trim(strtolower($string)));
        }

        $tokens = [];

        preg_match_all('/\w\w+/u', $string, $tokens);

        return $tokens[0];
    }
}
