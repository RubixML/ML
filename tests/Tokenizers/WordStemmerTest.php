<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Tokenizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\WordStemmer;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(WordStemmer::class)]
class WordStemmerTest extends TestCase
{
    protected WordStemmer $tokenizer;

    public static function tokenizeProvider() : Generator
    {
        /**
         * English
         */
        yield [
            "If something's important enough, you should try. Even if - the probable outcome is failure.",
            [
                'If', 'someth', 'import', 'enough', 'you', 'should', 'tri',
                'even', 'if', '-', 'the', 'probabl', 'outcom', 'is', 'failur',
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->tokenizer = new WordStemmer('english');
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
