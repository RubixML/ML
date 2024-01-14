<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\KSkipNGram;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\KSkipNGram
 */
class KSkipNGramTest extends TestCase
{
    /**
     * @var KSkipNGram
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new KSkipNGram(2, 3, 2);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(KSkipNGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     * @dataProvider tokenizeProvider
     *
     * @param string $text
     * @param list<string> $expected
     */
    public function tokenize(string $text, array $expected) : void
    {
        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function tokenizeProvider() : Generator
    {
        /**
         * English
         */
        yield [
            'I would like to die on Mars, just not on impact. The end.',
            [
                'I would', 'I like', 'I to', 'I would like', 'I like to', 'I to die', 'would like',
                'would to', 'would die', 'would like to', 'would to die', 'would die on', 'like to',
                'like die', 'like on', 'like to die', 'like die on', 'like on Mars', 'to die', 'to on',
                'to Mars', 'to die on', 'to on Mars', 'to Mars just', 'die on', 'die Mars', 'die just',
                'die on Mars', 'die Mars just', 'die just not', 'on Mars', 'on just', 'on not',
                'on Mars just', 'on just not', 'on not on', 'Mars just', 'Mars not', 'Mars on',
                'Mars just not', 'Mars not on', 'Mars on impact', 'just not', 'just on', 'just impact',
                'just not on', 'just on impact', 'not on', 'not impact', 'not on impact', 'on impact',
                'The end',
            ],
        ];
    }
}
