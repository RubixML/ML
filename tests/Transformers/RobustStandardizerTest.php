<?php

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\RobustStandardizer;
use PHPUnit\Framework\TestCase;

class RobustStandardizerTest extends TestCase
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

        $this->transformer = new RobustStandardizer();
    }

    public function test_build_z_scale_standardizer()
    {
        $this->assertInstanceOf(RobustStandardizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->transform($this->transformer);

        $this->assertEquals([
            [-0.9999999997435898, -0.9999999994444444, -0.9999999996296296, -0.9999999983333333],
            [0.0, 0.0, 0.0, 0.0],
            [1.5384615380670612, 15.55555554691358, 6.2962962939643345, 64.99999989166666],
        ], $this->dataset->samples());
    }
}
