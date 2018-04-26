<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\VarianceThresholdFilter;
use PHPUnit\Framework\TestCase;

class VarianceThresholdFilterTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Dataset([[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]);

        $this->transformer = new VarianceThresholdFilter(0.0);
    }

    public function test_build_variance_threshold_filter()
    {
        $this->assertInstanceOf(VarianceThresholdFilter::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->assertEquals([1, 2], $this->transformer->selected());
    }

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $data = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]];

        $this->transformer->transform($data);

        $this->assertEquals([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]], $data);
    }
}
