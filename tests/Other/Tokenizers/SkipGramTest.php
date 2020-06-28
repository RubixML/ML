<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\SkipGram;
use Rubix\ML\Other\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Other\Tokenizers\SkipGram
 */
class SkipGramTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\SkipGram
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new SkipGram(2, 2);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SkipGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I would', 'I like', 'I to', 'would like', 'would to', 'would die',
            'like to', 'like die', 'like on', 'to die', 'to on', 'to Mars', 'die on', 'die Mars',
            'die just', 'on Mars', 'on just', 'on not', 'Mars just', 'Mars not', 'Mars on',
            'just not', 'just on', 'just impact', 'not on', 'not impact', 'on impact', 'The end',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(28, $tokens);
    }
}
