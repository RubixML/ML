<?php

namespace Rubix\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class ZScaleStandardizerTest extends TestCase
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

        $this->transformer = new ZScaleStandardizer();
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(ZScaleStandardizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_fitted()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $this->assertEquals([
            [-1.1297063462715407, -0.772046361723632,  -0.8562475928233875, -0.7232368526547304],
            [-0.17191183534398832, -0.6401143885641433, -0.5466223472662737, -0.6908531130205375],
            [1.3016181815737315, 1.412160750241327, 1.4028699400276035, 1.4140899656383532],
        ], $this->dataset->samples());
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
