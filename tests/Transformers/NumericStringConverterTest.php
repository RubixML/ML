<?php

namespace Rubix\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\NumericStringConverter;
use PHPUnit\Framework\TestCase;

class NumericStringConverterTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            ['1', '2', '3', 4],
            ['4.0', '2.0', 3.0, 1.0],
            ['100', '3.0', '200', 1.0],
        ]);

        $this->transformer = new NumericStringConverter();
    }

    public function test_build_numeric_string_converter()
    {
        $this->assertInstanceOf(NumericStringConverter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_fitted()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->transform($this->transformer);

        $this->assertEquals([
            [1, 2, 3, 4],
            [4.0, 2.0, 3.0, 1.0],
            [100, 3.0, 200, 1.0],
        ], $this->dataset->samples());
    }
}
