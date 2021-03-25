<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\L2Normalizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\L2Normalizer
 */
class L2NormalizerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\L2Normalizer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled([
            [1, 2, 3, 4],
            [40, 0, 30, 10],
            [100, 300, 200, 400],
        ]);

        $this->transformer = new L2Normalizer();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(L2Normalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214],
            [0.7844645405527362, 0.0, 0.5883484054145521, 0.19611613513818404],
            [0.18257418583505536, 0.5477225575051661, 0.3651483716701107, 0.7302967433402214],
        ], $this->dataset->samples());
    }
}
