<?php

use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\MissingDataImputer;
use PHPUnit\Framework\TestCase;

class MissingDataImputerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
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
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->transform($this->transformer);

        $this->assertThat($this->dataset[0][1], $this->logicalAnd($this->greaterThan(30), $this->lessThan(45)));
        $this->assertContains($this->dataset[1][3], ['friendly', 'mean']);
    }
}
