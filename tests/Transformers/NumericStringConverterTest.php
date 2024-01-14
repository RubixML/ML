<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Reversible;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\NumericStringConverter;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\NumericStringConverter
 */
class NumericStringConverterTest extends TestCase
{
    /**
     * @var NumericStringConverter
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->transformer = new NumericStringConverter();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(NumericStringConverter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Reversible::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transformReverse() : void
    {
        $dataset = new Unlabeled([
            ['1', '2', 3, 4, 'NAN'],
            ['4.0', '2.0', 3.0, 1.0, 'INF'],
            ['100', '3.0', 200, 2.5, '-INF'],
        ]);

        $dataset->apply($this->transformer);

        $samples = $dataset->samples();

        $this->assertEquals(1, $samples[0][0]);
        $this->assertEquals(4.0, $samples[1][0]);
        $this->assertEquals(200, $samples[2][2]);
        $this->assertNan($samples[0][4]);
        $this->assertInfinite($samples[1][4]);
        $this->assertInfinite($samples[2][4]);

        $dataset->reverseApply($this->transformer);

        $expected = [
            ['1', '2', 3, 4, 'NAN'],
            ['4', '2', 3.0, 1.0, 'INF'],
            ['100', '3', 200, 2.5, '-INF'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
