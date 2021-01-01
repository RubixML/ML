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
     * @test
     */
    public function build() : void
    {
        $this->expectDeprecation();

        $tokenizer = new SkipGram(2, 2);

        $this->assertInstanceOf(SkipGram::class, $tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $this->expectDeprecation();

        $tokenizer = new SkipGram(2, 2);

        $text = 'I would like to die on Mars, just not on impact. The end.';

        $expected = [
            'I would', 'I like', 'I to', 'would like', 'would to', 'would die',
            'like to', 'like die', 'like on', 'to die', 'to on', 'to Mars', 'die on', 'die Mars',
            'die just', 'on Mars', 'on just', 'on not', 'Mars just', 'Mars not', 'Mars on',
            'just not', 'just on', 'just impact', 'not on', 'not impact', 'on impact', 'The end',
        ];

        $tokens = $tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(28, $tokens);
    }
}
