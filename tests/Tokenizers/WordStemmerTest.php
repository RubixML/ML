<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Tokenizers\Tokenizer;
use Rubix\ML\Tokenizers\WordStemmer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\WordStemmer
 */
class WordStemmerTest extends TestCase
{
    /**
     * @var WordStemmer
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new WordStemmer('english');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(WordStemmer::class, $this->tokenizer);
        $this->assertInstanceOf(Word::class, $this->tokenizer);
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
            "If something's important enough, you should try. Even if - the probable outcome is failure.",
            [
                'If', 'someth', 'import', 'enough', 'you', 'should', 'tri',
                'even', 'if', '-', 'the', 'probabl', 'outcom', 'is', 'failur',
            ],
        ];
    }
}
