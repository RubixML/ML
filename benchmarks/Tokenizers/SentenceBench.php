<?php

namespace Rubix\ML\Benchmarks\Tokenizers;

use Rubix\ML\Tokenizers\Sentence;

/**
 * @Groups({"Tokenizers"})
 * @BeforeMethods({"setUp"})
 */
class SentenceBench
{
    protected const SAMPLE_TEXT = "Do you see any Teletubbies in here? Do you see a slender plastic tag clipped to my shirt with my name printed on it? Do you see a little Asian child with a blank expression on his face sitting outside on a mechanical helicopter that shakes when you put quarters in it? No? Well, that's what you see at a toy store. And you must think you're in a toy store, because you're here shopping for an infant named Jeb.";

    /**
     * @var \Rubix\ML\Tokenizers\Sentence;
     */
    protected $tokenizer;

    public function setUp() : void
    {
        $this->tokenizer = new Sentence();
    }

    /**
     * @Subject
     * @revs(30)
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function tokenize() : void
    {
        $this->tokenizer->tokenize(self::SAMPLE_TEXT);
    }
}
