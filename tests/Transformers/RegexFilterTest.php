<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\RegexFilter;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(RegexFilter::class)]
class RegexFilterTest extends TestCase
{
    protected RegexFilter $transformer;

    protected function setUp() : void
    {
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

    public function testTransform() : void
    {
        $dataset = Unlabeled::quick([
            ['I was not proud of what I had learned, but I never doubted that it was worth $$$ knowing..'],
            ['Too weird to live, support@rubixml.com too rare to die https://rubixml.com'],
            ['A man who procrastinates in @his choosing will inevitably have his choice    made for him by #circumstance'],
            ['The quick quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of Cola cola'],
            ['Diese₂ äpfel Äpfel schmecken sehr gut'],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            ['I was not proud of what I had learned, but I never doubted that it was worth $ knowing.'],
            ['Too weird to live, too rare to die '],
            ['A man who procrastinates in choosing will inevitably have his choice made for him by '],
            ['The quick brown fox jumped over the lazy man sitting at a bus stop drinking a can of cola'],
            ['Diese₂ Äpfel schmecken sehr gut'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
