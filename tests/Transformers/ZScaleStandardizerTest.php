<?php

use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use PHPUnit\Framework\TestCase;

class ZScaleStandardizerTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->transformer = new ZScaleStandardizer();

        $this->transformer->fit(new Unlabeled([
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]));
    }

    public function test_build_z_scale_standardizer()
    {
        $this->assertInstanceOf(ZScaleStandardizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            [-1.1297063462715407, -0.772046361723632,  -0.8562475928233875, -0.7232368526547304],
            [-0.17191183534398832, -0.6401143885641433, -0.5466223472662737, -0.6908531130205375],
            [1.3016181815737315, 1.412160750241327, 1.4028699400276035, 1.4140899656383532],
        ], $data);
    }
}
