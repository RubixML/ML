<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Sentence;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Other\Tokenizers\Sentence
 */
class SentenceTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\Sentence
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new Sentence();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Sentence::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = "I'd like to die on Mars, just not on-impact. The end.";

        $expected = [
            "I'd like to die on Mars, just not on-impact.",
            'The end.',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(2, $tokens);
    }
}
