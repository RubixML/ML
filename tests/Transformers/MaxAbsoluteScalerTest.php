<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\MaxAbsoluteScaler;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class MaxAbsoluteScalerTest extends TestCase
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

        $this->transformer = new MaxAbsoluteScaler();
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(MaxAbsoluteScaler::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [0.01, 0.006666666666666667,  0.015, 0.01],
            [0.4, 0.06666666666666667, 0.15, 0.025],
            [1., 1., 1., 1.],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
