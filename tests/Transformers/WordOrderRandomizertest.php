<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Transformers\WordOrderRandomizer;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\WordOrderRandomizer
 */
class WordOrderRandomizertest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\TextNormalizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['Red dining chair.'],
            ['Blue,cotton,pillow'],
        ]);

        $this->transformer = new WordOrderRandomizer();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(WordOrderRandomizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        foreach (explode(' ', 'Red dining chair.') as $word) {
            $this->assertTrue(str_contains($this->dataset->samples()[0][0], $word));
        }

        $this->assertEquals(['Blue,cotton,pillow'], $this->dataset->samples()[1]);
    }
}
