<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\MissingDataImputer;
use PHPUnit\Framework\TestCase;

class MissingDataImputerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Dataset([
            [30, 'friendly'],
            ['?', 'mean'],
            [50, 'friendly'],
            [60, '?'],
            [10, 'mean'],
        ]);

        $this->transformer = new MissingDataImputer('?');
    }

    public function test_build_imputer()
    {
        $this->assertInstanceOf(MissingDataImputer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['?', '?'],
        ];

        $this->transformer->fit($this->dataset);

        $this->transformer->transform($data);

        $this->assertThat($data[0][0], $this->logicalAnd($this->greaterThan(30), $this->lessThan(45)));
        $this->assertContains($data[0][1], ['friendly', 'mean']);
    }
}
