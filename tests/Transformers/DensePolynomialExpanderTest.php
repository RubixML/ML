<?php

use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\DensePolynomialExpander;
use PHPUnit\Framework\TestCase;

class DensePolynomialExpanderTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]);

        $this->transformer = new DensePolynomialExpander(2, true);
    }

    public function test_build_l1_regularizer()
    {
        $this->assertInstanceOf(DensePolynomialExpander::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->transform($this->transformer);

        $this->assertEquals([
            [1, 1, 2, 4, 3, 9, 4, 16],
            [40, 1600, 20, 400, 30, 900, 10, 100],
            [100, 10000, 300, 90000, 200, 40000, 400, 160000],
        ], $this->dataset->samples());
    }
}
