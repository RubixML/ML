<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Tokenizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\KSkipNGram;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(KSkipNGram::class)]
class KSkipNGramTest extends TestCase
{
    protected KSkipNGram $tokenizer;

    /**
     * @return Generator<mixed[]>
     */
    public static function tokenizeProvider() : Generator
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

    /**
     * @return Generator<mixed[]>
     */
    public static function tokenizeUnigramProvider() : Generator
    {
        /**
         * English
         */
        yield [
            'I would like to die on Mars, just not on impact. The end.',
            [
                'I', 'would', 'like', 'to', 'die', 'on', 'Mars', 'just', 'not', 'on', 'impact',
                'The', 'end',
            ],
        ];
    }

    /**
     * @return Generator<mixed[]>
     */
    public static function minThreeProvider() : Generator
    {
        yield [
            'a b c d e',
            [
                'a b c', 'a c d', 'a b c d', 'a c d e', 'b c d',
                'b d e', 'b c d e', 'c d e',
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->tokenizer = new KSkipNGram(min: 2, max: 3, skip: 2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(KSkipNGram::class, $this->tokenizer);
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
    #[DataProvider('tokenizeUnigramProvider')]
    #[Test]
    public function tokenizeUnigrams(string $text, array $expected) : void
    {
        $tokenizer = new KSkipNGram(min: 1, max: 1, skip: 2);

        $tokens = $tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }

    /**
     * @param string $text
     * @param list<string> $expected
     */
    #[DataProvider('minThreeProvider')]
    #[Test]
    public function tokenizeMinThree(string $text, array $expected) : void
    {
        $tokenizer = new KSkipNGram(min: 3, max: 4, skip: 1);

        $tokens = $tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }
}
