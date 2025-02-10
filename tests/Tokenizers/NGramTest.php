<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Tokenizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\NGram;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(NGram::class)]
class NGramTest extends TestCase
{
    protected NGram $tokenizer;

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

    protected function setUp() : void
    {
        $this->tokenizer = new NGram(min: 1, max: 2);
    }

    /**
     * @param string $text
     * @param list<string> $expected
     */
    #[DataProvider('tokenizeProvider')]
    public function testTokenize(string $text, array $expected) : void
    {
        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
    }
}
