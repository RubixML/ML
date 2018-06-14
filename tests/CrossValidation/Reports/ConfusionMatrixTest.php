<?php

use Rubix\Engine\Datasets\Labeled;
use Rubix\Tests\Helpers\MockClassifier;
use Rubix\Engine\CrossValidation\Reports\Report;
use Rubix\Engine\CrossValidation\Reports\Classification;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use PHPUnit\Framework\TestCase;

class ConfusionMatrixTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = new MockClassifier([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf'
        ]);

        $this->report = new ConfusionMatrix(['wolf', 'lamb']);
    }

    public function test_build_confusion_matrix()
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(Classification::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $actual = [
            'wolf' => [
                'wolf' => 2,
                'lamb' => 1,
            ],
            'lamb' => [
                'wolf' => 1,
                'lamb' => 1,
            ],
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($actual, $result);
    }
}
