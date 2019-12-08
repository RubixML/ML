<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class WordTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\Word
     */
    protected $tokenizer;

    public function setUp() : void
    {
        $this->tokenizer = new Word();
    }

    public function test_build_tokenizer() : void
    {
        $this->assertInstanceOf(Word::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I', 'would', 'like', 'to', 'die', 'on', 'Mars', 'just', 'not', 'on', 'impact',
            'The', 'end',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertCount(13, $tokens);
        $this->assertEquals($expected, $tokens);
    }
}
