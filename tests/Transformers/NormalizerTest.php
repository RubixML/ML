<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Normalizer;
use PHPUnit\Framework\TestCase;

class NormalizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new Normalizer();

        $this->preprocessor->fit(new Dataset([
            [50, 100, 1000],
            [40, 200, 3000],
            [29, 300, 2000],
        ]));
    }

    public function test_build_z_scale_standardizer()
    {
        $this->assertInstanceOf(Normalizer::class, $this->preprocessor);
    }

    public function test_fit_dataset()
    {
        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            [50, 100, 1000],
            [40, 200, 3000],
            [29, 247, 1999],
        ];

        $this->preprocessor->transform($data);

        $this->assertEquals([
            [1, 0, 0],
            [0.5238095238095238, 0.5, 1],
            [0, 0.735, 0.4995],
        ], $data);
    }
}
