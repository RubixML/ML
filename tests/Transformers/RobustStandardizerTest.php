<?php

namespace Rubix\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\RobustStandardizer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

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

    public function test_build_transformer()
    {
        $this->assertInstanceOf(RobustStandardizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_fitted()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [-0.6744999998270512, -0.6744999996252777, -0.6744999997501852, -0.6744999988758333],
            [0.0, 0.0, 0.0, 0.0],
            [1.0376923074262328, 10.492222216393209, 4.246851850278944, 43.84249992692917],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
