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
            [-0.6744999996541025, -0.6744999992505555, -0.6744999995003703, -0.6744999977516667],
            [0.0, 0.0, 0.0, 0.0],
            [1.0376923071601578, 10.492222210564195, 4.246851848706035, 43.842499853858335],
        ], $this->dataset->samples());
    }
}
