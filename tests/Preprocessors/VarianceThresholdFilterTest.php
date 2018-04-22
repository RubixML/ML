<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Transformers\VarianceThresholdFilter;
use PHPUnit\Framework\TestCase;

class VarianceThresholdFilterTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new VarianceThresholdFilter(0.2);
    }

    public function test_build_variance_threshold_selector()
    {
        $this->assertInstanceOf(VarianceThresholdFilter::class, $this->preprocessor);
    }

    public function test_fit_dataset()
    {
        $data = new Dataset([[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]);

        $this->preprocessor->fit($data);

        $this->assertEquals([1, 2], $this->preprocessor->selected());
    }

    public function test_transform_dataset()
    {
        $data = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]];

        $this->preprocessor->transform($data);

        $this->assertEquals($data = [[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]], $data);
    }
}
