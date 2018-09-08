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
            [-1., -1., -1., -1.],
            [0.0, 0.0, 0.0, 0.0],
            [1.5384615384615385, 15.555555555555555, 6.296296296296297, 65.],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
