<?php

namespace Rubix\ML\Tests\Reports;

use Rubix\ML\Reports\Report;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\Classifiers\KNearestNeighbors;
use PHPUnit\Framework\TestCase;

class MulticlassBreakdownTest extends TestCase
{
    protected $report;

    protected $testing;

    protected $estimator;

    public function setUp()
    {
        $samples = [[], [], [], [], []];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $this->testing = Labeled::quick($samples, $labels);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

        $this->estimator->method('type')->willReturn(KNearestNeighbors::CLASSIFIER);

        $this->estimator->method('predict')->willReturn([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf',
        ]);

        $this->report = new MulticlassBreakdown();
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(MulticlassBreakdown::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $outcome = [
            'overall' => [
                'accuracy' => 0.6,
                'precision' => 0.5833333333333333,
                'recall' => 0.5833333333333333,
                'specificity' => 0.5833333333333333,
                'miss_rate' => 0.4166666666666667,
                'fall_out' => 0.4166666666666667,
                'f1_score' => 0.5833333333333333,
                'informedness' => 0.16666666666666652,
                'mcc' => 0.16666666666666666,
            ],
            'label' => [
                'wolf' => [
                    'accuracy' => 0.6,
                    'precision' => 0.6666666666666666,
                    'recall' => 0.6666666666666666,
                    'specificity' => 0.5,
                    'miss_rate' => 0.33333333333333337,
                    'fall_out' => 0.5,
                    'f1_score' => 0.6666666666666666,
                    'informedness' => 0.16666666666666652,
                    'mcc' => 0.16666666666666666,
                    'cardinality' => 3,
                    'density' => 0.6,
                    'true_positives' => 2,
                    'true_negatives' => 1,
                    'false_positives' => 1,
                    'false_negatives' => 1,
                ],
                'lamb' => [
                    'accuracy' => 0.6,
                    'precision' => 0.5,
                    'recall' => 0.5,
                    'specificity' => 0.6666666666666666,
                    'miss_rate' => 0.5,
                    'fall_out' => 0.33333333333333337,
                    'f1_score' => 0.5,
                    'informedness' => 0.16666666666666652,
                    'mcc' => 0.16666666666666666,
                    'cardinality' => 2,
                    'density' => 0.4,
                    'true_positives' => 1,
                    'true_negatives' => 2,
                    'false_positives' => 1,
                    'false_negatives' => 1,
                ],
            ],
        ];

        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($outcome, $result);
    }
}
