<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Tokenizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\NGram;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(NGram::class)]
class NGramTest extends TestCase
{
    protected NGram $tokenizer;

    /**
     * @return Generator<mixed[]>
     */
    public static function tokenizeProvider() : Generator
    {
        /**
         * English
         */
        yield [
            "I'd like to die on Mars, just not on impact. The end.",
            [
                "I'd", "I'd like", 'like', 'like to', 'to', 'to die', 'die',
                'die on', 'on', 'on Mars', 'Mars', 'Mars just', 'just', 'just not', 'not', 'not on',
                'on', 'on impact', 'impact', 'The', 'The end', 'end',
            ],
        ];
    }

    /**
     * @return Generator<mixed[]>
     */
    public static function trigramProvider() : Generator
    {
        yield [
            'the quick brown fox jumps',
            [
                'the quick', 'the quick brown', 'quick brown', 'quick brown fox',
                'brown fox', 'brown fox jumps', 'fox jumps',
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->tokenizer = new NGram(min: 1, max: 2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(NGram::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @param string $text
     * @param list<string> $expected
     */
    #[DataProvider('tokenizeProvider')]
    #[Test]
    public function tokenize(string $text, array $expected) : void
    {
        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }

    /**
     * @param string $text
     * @param list<string> $expected
     */
    #[DataProvider('trigramProvider')]
    #[Test]
    public function tokenizeTrigrams(string $text, array $expected) : void
    {
        $tokenizer = new NGram(min: 2, max: 3);

        $tokens = $tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }
}
