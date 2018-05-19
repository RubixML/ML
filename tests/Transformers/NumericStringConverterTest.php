<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\NumericStringConverter;
use PHPUnit\Framework\TestCase;

class NumericStringConverterTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->transformer = new NumericStringConverter();

        $this->transformer->fit(new Dataset([[1, 2, 3, 4]]));
    }

    public function test_build_l1_regularizer()
    {
        $this->assertInstanceOf(NumericStringConverter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['1', '2', '3', 4],
            ['4.0', '2.0', 3.0, 1.0],
            ['100', '3.0', '200', 1.0],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            [1, 2, 3, 4],
            [4.0, 2.0, 3.0, 1.0],
            [100, 3.0, 200, 1.0],
        ], $data);
    }
}
