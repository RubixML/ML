<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\Word
 */
class WordTest extends TestCase
{
    /**
     * @var \Rubix\ML\Tokenizers\Word
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new Word();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Word::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = "I'd like to die on Mars, just not on-impact. The end.";

        $expected = [
            "I'd", 'like', 'to', 'die', 'on', 'Mars', 'just', 'not', 'on-impact',
            'The', 'end',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(11, $tokens);
    }
}
