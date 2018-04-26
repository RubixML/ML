<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\MissingDataImputer;
use PHPUnit\Framework\TestCase;

class MissingDataImputerTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->transformer = new MissingDataImputer('?');

        $this->transformer->fit(new Dataset([
            [30, 'friendly'],
            [40, 'mean'],
            [50, 'friendly'],
        ]));
    }

    public function test_build_imputer()
    {
        $this->assertInstanceOf(MissingDataImputer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['?', '?'],
        ];

        $this->transformer->transform($data);

        $this->assertTrue($data[0][0] > 37 && $data[0][0] < 43);
        $this->assertTrue(in_array($data[0][1], ['friendly', 'mean']));
    }
}
