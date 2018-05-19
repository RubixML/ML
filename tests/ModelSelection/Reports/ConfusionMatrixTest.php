<?php

use Rubix\Engine\ModelSelection\Reports\Report;
use Rubix\Engine\Estimators\Predictions\Prediction;
use Rubix\Engine\ModelSelection\Reports\ConfusionMatrix;
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
        $predictions = [
            new Prediction('wolf'), new Prediction('lamb'),
            new Prediction('wolf'), new Prediction('lamb'),
            new Prediction('wolf'),
        ];

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
