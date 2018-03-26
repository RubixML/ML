<?php

use Rubix\Engine\Preprocessors\L2Normalizer;
use PHPUnit\Framework\TestCase;

class L2NormalizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new L2Normalizer();
    }

    public function test_build_l1_normalizer()
    {
        $this->assertInstanceOf(L2Normalizer::class, $this->preprocessor);
    }

    public function test_fit_dataset()
    {
        $this->preprocessor->fit([]);

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
            [0.31622776601683794, 0.6324555320336759, 0.9486832980505138, 1.2649110640673518],
            [4.0, 2.0, 3.0, 1.0],
            [3.1622776601683795, 9.486832980505138, 6.324555320336759, 12.649110640673518],
        ], $data);
    }
}
