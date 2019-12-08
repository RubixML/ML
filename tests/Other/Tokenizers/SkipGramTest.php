<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\SkipGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

class SkipGramTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\SkipGram
     */
    protected $tokenizer;

    public function setUp() : void
    {
        $this->tokenizer = new SkipGram(2, 2);
    }

    public function test_build_tokenizer() : void
    {
        $this->assertInstanceOf(SkipGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    public function test_tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I would', 'I like', 'I to', 'would like', 'would to', 'would die',
            'like to', 'like die', 'like on', 'to die', 'to on', 'to Mars', 'die on', 'die Mars',
            'die just', 'on Mars', 'on just', 'on not', 'Mars just', 'Mars not', 'Mars on',
            'just not', 'just on', 'just impact', 'on impact', 'The end',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertCount(26, $tokens);
        $this->assertEquals($expected, $tokens);
    }
}
