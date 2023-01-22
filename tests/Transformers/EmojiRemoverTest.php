<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\EmojiRemover;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\EmojiRemover
 */
class EmojiRemoverTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\EmojiRemover
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['The quick 😀 brown 🦊 jumped over the lazy 🛌 man sitting at a bus stop 🚍 drinking a can of 🥤'],
            ['with a Dandy 🌂 umbrella 🌂'],
            ["don't touch this one!!@$%^&"],
        ]);

        $this->transformer = new EmojiRemover();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(EmojiRemover::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $outcome = [
            ['The quick  brown  jumped over the lazy  man sitting at a bus stop  drinking a can of '],
            ['with a Dandy  umbrella '],
            ["don't touch this one!!@$%^&"],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }
}