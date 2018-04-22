<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Transformers\L2Regularizer;
use PHPUnit\Framework\TestCase;

class L2RegularizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new L2Regularizer();

        $this->preprocessor->fit(new Dataset([[1, 2, 3, 4]]));
    }

    public function test_build_l1_regularizer()
    {
        $this->assertInstanceOf(L2Regularizer::class, $this->preprocessor);
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
            [0.18257418583172202, 0.36514837166344405, 0.5477225574951661, 0.7302967433268881],
            [0.730296743338888, 0.365148371669444, 0.5477225575041661, 0.182574185834722],
            [0.18257418583502202, 0.547722557505066, 0.36514837167004405, 0.7302967433400881],
        ], $data);
    }
}
