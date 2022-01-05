<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\InfNanConverter;
use Rubix\ML\Transformers\Reversible;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\InfNanConverter
 */
class InfNanConverterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\InfNanConverter
     */
    protected $transformer;

    /**
     * @var array<mixed[]>
     */
    protected $samples = [
        [1, 'abc', true, INF, -INF, NAN],
        [2, 'def', false, -INF, NAN, 0],
    ];

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled($this->samples);
        $this->transformer = new InfNanConverter();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(InfNanConverter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Reversible::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [1, 'abc', true, '~~INF~~', '~~-INF~~', '~~NAN~~'],
            [2, 'def', false, '~~-INF~~', '~~NAN~~', 0],
        ], $this->dataset->samples());
    }

    /**
     * @test
     */
    public function reverseTransform() : void
    {
        $this->dataset->apply($this->transformer);
        $this->dataset->reverseApply($this->transformer);

        // Can't compare array due to NAN values
        $this->assertEquals(1, $this->dataset->samples()[0][0]);
        $this->assertEquals('abc', $this->dataset->samples()[0][1]);
        $this->assertEquals(true, $this->dataset->samples()[0][2]);
        $this->assertEquals(INF, $this->dataset->samples()[0][3]);
        $this->assertEquals(-INF, $this->dataset->samples()[0][4]);
        $this->assertTrue(is_nan($this->dataset->samples()[0][5]));

        $this->assertEquals(2, $this->dataset->samples()[1][0]);
        $this->assertEquals('def', $this->dataset->samples()[1][1]);
        $this->assertEquals(false, $this->dataset->samples()[1][2]);
        $this->assertEquals(-INF, $this->dataset->samples()[1][3]);
        $this->assertTrue(is_nan($this->dataset->samples()[1][4]));
        $this->assertEquals(0, $this->dataset->samples()[1][5]);
    }
}
