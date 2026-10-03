<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Tokenizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tokenizers\Stemmers\PorterEnglish;
use Rubix\ML\Tokenizers\Stemmers\Stemmer;
use Rubix\ML\Tokenizers\Tokenizer;
use Rubix\ML\Tokenizers\WordStemmer;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Tokenizers')]
#[CoversClass(WordStemmer::class)]
class WordStemmerTest extends TestCase
{
    protected WordStemmer $tokenizer;

    /**
     * @return Generator<mixed[]>
     */
    public static function tokenizeProvider() : Generator
    {
        yield [
            'Majority voting is likely foolish',
            ['Major', 'vote', 'is', 'like', 'foolish'],
        ];

        yield [
            'Running and jumping are fun exercises',
            ['Run', 'and', 'jump', 'ar', 'fun', 'exercis'],
        ];

        yield [
            'I like to eat apples and oranges',
            ['I', 'like', 'to', 'eat', 'appl', 'and', 'orang'],
        ];
    }

    protected function setUp() : void
    {
        $this->tokenizer = new WordStemmer();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(WordStemmer::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
        $this->assertInstanceOf(Stemmer::class, $this->tokenizer->stemmer());
        $this->assertInstanceOf(PorterEnglish::class, $this->tokenizer->stemmer());
    }

    #[Test]
    public function useCustomStemmer() : void
    {
        $stemmer = new PorterEnglish();

        $tokenizer = new WordStemmer($stemmer);

        $this->assertSame($stemmer, $tokenizer->stemmer());
        $this->assertSame(
            ['Caress', 'poni'],
            $tokenizer->tokenize('Caressing ponies'),
        );
    }

    /**
     * @param string $text
     * @param list<string> $expected
     */
    #[DataProvider('tokenizeProvider')]
    #[Test]
    public function tokenize(string $text, array $expected) : void
    {
        $this->assertSame($expected, $this->tokenizer->tokenize($text));
    }

    #[Test]
    public function tokenizeIsRepeatable() : void
    {
        $this->assertSame(
            ['Major', 'vote', 'is', 'like', 'foolish'],
            $this->tokenizer->tokenize('Majority voting is likely foolish'),
        );

        $this->assertSame(
            ['Major', 'vote', 'is', 'like', 'foolish'],
            $this->tokenizer->tokenize('Majority voting is likely foolish'),
        );
    }

    #[Test]
    public function toStringReturnsTokenizer() : void
    {
        $this->assertSame('Word Stemmer (language: PorterEnglish)', (string) $this->tokenizer);
    }
}
