<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\RegexFilter;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\RegexFilter
 */
class RegexFilterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\RegexFilter
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['I was not proud of what I had learned, but I never doubted that it was worth $$$ knowing..'],
            ['Too weird to live, support@rubixml.com too rare to die https://rubixml.com'],
            ['A man who procrastinates in @his choosing will inevitably have his choice    made for him by #circumstance'],
            ['The quick quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of Cola cola'],
            ['Diese äpfel Äpfel schmecken sehr gut'],
            ['The quick 😀 brown 🦊 jumped over the lazy 🛌 man sitting at a bus stop 🚍 drinking a can of 🥤']
            ['Diese₂ äpfel Äpfel schmecken sehr gut'],
        ]);

        $this->transformer = new RegexFilter([
            RegexFilter::URL,
            RegexFilter::EMAIL,
            RegexFilter::EXTRA_CHARACTERS,
            RegexFilter::EXTRA_WORDS,
            RegexFilter::MENTION,
            RegexFilter::HASHTAG,
            RegexFilter::EXTRA_WHITESPACE,
            RegexFilter::EMOJIS,
        ]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RegexFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $expected = [
            ['I was not proud of what I had learned, but I never doubted that it was worth $ knowing.'],
            ['Too weird to live, too rare to die '],
            ['A man who procrastinates in choosing will inevitably have his choice made for him by '],
            ['The quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of cola'],
            ['The quick  brown  jumped over the lazy  man sitting at a bus stop  drinking a can of '],
            ['Diese₂ Äpfel schmecken sehr gut'],
        ];

        $this->assertEquals($expected, $this->dataset->samples());
    }
}
