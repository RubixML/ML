<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\PolynomialExpander;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\PolynomialExpander
 */
class PolynomialExpanderTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\PolynomialExpander
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled([
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]);

        $this->transformer = new PolynomialExpander(2);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(PolynomialExpander::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $outcome = [
            [1, 1, 2, 4, 3, 9, 4, 16],
            [40, 1600, 20, 400, 30, 900, 10, 100],
            [100, 10000, 300, 90000, 200, 40000, 400, 160000],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
