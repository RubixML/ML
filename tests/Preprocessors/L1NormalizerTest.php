<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Preprocessors\L1Normalizer;
use PHPUnit\Framework\TestCase;

class L1NormalizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new L1Normalizer();

        $this->preprocessor->fit(new Dataset([[1, 2, 3, 4]]));
    }

    public function test_build_l1_normalizer()
    {
        $this->assertInstanceOf(L1Normalizer::class, $this->preprocessor);
    }

    public function test_fit_dataset()
    {
        $this->assertEquals([0, 1, 2, 3], $this->preprocessor->columns());
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
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.2, 0.3, 0.1],
            [0.1, 0.3, 0.2, 0.4],
        ], $data);
    }
}
