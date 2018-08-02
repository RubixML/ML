<?php

namespace Rubix\Tests\Reports;

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
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

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
        $actual = [
            'overall' => [
                'accuracy' => 0.6,
                'precision' => 0.5833333309722222,
                'recall' => 0.5833333309722222,
                'specificity' => 0.5833333309722222,
                'miss_rate' => 0.41666666902777777,
                'fall_out' => 0.41666666902777777,
                'f1_score' => 0.5833333259722223,
                'informedness' => 0.16666666194444435,
                'mcc' => 0.16666666638888888,
            ],
            'label' => [
                'wolf' => [
                    'accuracy' => 0.6,
                    'precision' => 0.6666666644444444,
                    'recall' => 0.6666666644444444,
                    'specificity' => 0.4999999975,
                    'miss_rate' => 0.33333333555555555,
                    'fall_out' => 0.5000000025,
                    'f1_score' => 0.6666666594444445,
                    'informedness' => 0.16666666194444435,
                    'mcc' => 0.16666666638888888,
                    'cardinality' => 3,
                    'density' => 0.6,
                    'true_positives' => 2,
                    'true_negatives' => 1,
                    'false_positives' => 1,
                    'false_negatives' => 1,
                ],
                'lamb' => [
                    'accuracy' => 0.6,
                    'precision' => 0.4999999975,
                    'recall' => 0.4999999975,
                    'specificity' => 0.6666666644444444,
                    'miss_rate' => 0.5000000025,
                    'fall_out' => 0.33333333555555555,
                    'f1_score' => 0.4999999925000001,
                    'informedness' => 0.16666666194444435,
                    'mcc' => 0.16666666638888888,
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

        $this->assertEquals($actual, $result);
    }
}
