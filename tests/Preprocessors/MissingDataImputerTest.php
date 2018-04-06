<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Preprocessors\MissingDataImputer;
use PHPUnit\Framework\TestCase;

class MissingDataImputerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new MissingDataImputer('?');

        $this->preprocessor->fit(new Dataset([
            [30, 'friendly'],
            [40, 'mean'],
            [50, 'friendly'],
        ]));
    }

    public function test_build_imputer()
    {
        $this->assertInstanceOf(MissingDataImputer::class, $this->preprocessor);
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

        $this->preprocessor->transform($data);

        $this->assertTrue($data[0][0] > 39 && $data[0][0] < 41);
        $this->assertTrue(in_array($data[0][1], ['friendly', 'mean']));
    }
}
