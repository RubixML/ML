<?php

use Rubix\Engine\CrossValidation\Reports\Report;
use Rubix\Engine\CrossValidation\Reports\ConfusionMatrix;
use PHPUnit\Framework\TestCase;

class ConfusionMatrixTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->report = new ConfusionMatrix(['wolf', 'lamb']);
    }

    public function test_build_confusion_matrix()
    {
        $this->assertInstanceOf(ConfusionMatrix::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $matrix = [
            'wolf' => [
                'wolf' => 2,
                'lamb' => 1,
            ],
            'lamb' => [
                'wolf' => 1,
                'lamb' => 1,
            ],
        ];

        $report = $this->report->generate($predictions, $labels);

        $this->assertEquals($matrix, $report);
    }
}
