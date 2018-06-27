<?php

namespace Rubix\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\VarianceThresholdFilter;
use PHPUnit\Framework\TestCase;

class VarianceThresholdFilterTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [0, 0, 1], [0, 1, 0], [0, 0, 0],
            [0, 1, 1], [0, 1, 0], [0, 1, 1]
        ]);

        $this->transformer = new VarianceThresholdFilter(0.0);
    }

    public function test_build_variance_threshold_filter()
    {
        $this->assertInstanceOf(VarianceThresholdFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->transform($this->transformer);

        $this->assertEquals([
            [0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1]
        ], $this->dataset->samples());
    }
}
