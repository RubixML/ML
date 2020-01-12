<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\NumericStringConverter;
use PHPUnit\Framework\TestCase;

class NumericStringConverterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\NumericStringConverter
     */
    protected $transformer;

    public function setUp() : void
    {
        $this->dataset = new Unlabeled([
            ['1', '2', '3', 4],
            ['4.0', '2.0', '3.0', 1.0],
            ['100', '3.0', '200', 2.5],
        ]);

        $this->transformer = new NumericStringConverter();
    }

    public function test_build_transformer() : void
    {
        $this->assertInstanceOf(NumericStringConverter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform() : void
    {
        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [1, 2, 3, 4],
            [4.0, 2.0, 3.0, 1.0],
            [100, 3.0, 200, 2.5],
        ], $this->dataset->samples());
    }
}
