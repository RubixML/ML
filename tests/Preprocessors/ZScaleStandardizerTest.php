<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Transformers\ZScaleStandardizer;
use PHPUnit\Framework\TestCase;

class ZScaleStandardizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new ZScaleStandardizer();

        $this->preprocessor->fit(new Dataset([
            [1, 2, 3, 4],
            [40, 20, 30, 10],
            [100, 300, 200, 400],
        ]));
    }

    public function test_build_z_scale_standardizer()
    {
        $this->assertInstanceOf(ZScaleStandardizer::class, $this->preprocessor);
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

        $this->preprocessor->transform($data);

        $this->assertEquals([
            [-1.129706346546209, -0.7720463617796538, -0.8562475929205965, -0.7232368526933752],
            [-0.17191183534398832, -0.6401143885641433, -0.5466223472662737, -0.6908531130205375],
            [1.3016181818901973, 1.4121607503437972, 1.40286994018687, 1.4140899657139128],
        ], $data);
    }
}
