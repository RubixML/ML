<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\L1Normalizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\L1Normalizer
 */
class L1NormalizerTest extends TestCase
{
    /**
     * @var L1Normalizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->transformer = new L1Normalizer();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(L1Normalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $dataset = new Unlabeled([
            [1, 2, 3, 4],
            [40, 0, 30, 10],
            [100, 300, 200, 400],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.0, 0.375, 0.125],
            [0.1, 0.3, 0.2, 0.4],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
