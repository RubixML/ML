<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Sentence;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\Sentence
 */
class SentenceTest extends TestCase
{
    /**
     * @var \Rubix\ML\Tokenizers\Sentence
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new Sentence();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Sentence::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     * @dataProvider tokenizeProvider
     */
    public function tokenize(string $text, array $expected) : void
    {
        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(count($expected), $tokens);
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
            "I canceled my meeting with friends today because I\'m too tired! After several months, I\'m going to see my family again. Will that be good for me?Can you help me?\nThis book is very interesting! \"Given the circumstances we have now, can we make a meeting appointment\"?",
            [
                "I canceled my meeting with friends today because I\'m too tired!",
                "After several months, I\'m going to see my family again.",
                "Will that be good for me?",
                "Can you help me?",
                "This book is very interesting!",
                "\"Given the circumstances we have now, can we make a meeting appointment\"?"
            ]
        ];

        /**
         * Farsi
         */
        yield [
            "من امروز ملاقات با دوستانم را لغو کردم، چراکه خیلی خسته هستم! بعد از چند ماه مجدداً به دیدار خانواده‌ام می‌روم. آیا این برای من خوب خواهد بود؟آیا توانستی به من کمک کنی؟\nاین کتاب بسیار جالب است! \"با توجه به شرایطی که الان داریم، آیا می‌توانیم به یک قرار ملاقات برسیم\"؟",
            [
                "من امروز ملاقات با دوستانم را لغو کردم، چراکه خیلی خسته هستم!",
                "بعد از چند ماه مجدداً به دیدار خانواده‌ام می‌روم.",
                "آیا این برای من خوب خواهد بود؟",
                "آیا توانستی به من کمک کنی؟",
                "این کتاب بسیار جالب است!",
                "\"با توجه به شرایطی که الان داریم، آیا می‌توانیم به یک قرار ملاقات برسیم\"؟"
            ]
        ];
    }
}
