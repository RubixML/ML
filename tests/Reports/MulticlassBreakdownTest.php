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

    protected $outcome;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

        $this->estimator = $this->createMock(KNearestNeighbors::class);

        $this->estimator->method('predict')->willReturn([
            'wolf', 'lamb', 'wolf', 'lamb', 'wolf',
        ]);

        $this->report = new MulticlassBreakdown();

        $this->outcome = [
            'overall' => [
                'accuracy' => 0.6,
                'precision' => 0.5833333351388889,
                'recall' => 0.5833333351388889,
                'specificity' => 0.5833333351388889,
                'miss_rate' => 0.41666666486111115,
                'fall_out' => 0.41666666486111115,
                'f1_score' => 0.5833333476388887,
                'informedness' => 0.1666666702777777,
                'mcc' => 0.16666666805555555,
            ],
            'label' => [
                'wolf' => [
                    'accuracy' => 0.6,
                    'precision' => 0.6666666677777777,
                    'recall' => 0.6666666677777777,
                    'specificity' => 0.5000000025,
                    'miss_rate' => 0.3333333322222223,
                    'fall_out' => 0.4999999975,
                    'f1_score' => 0.6666666777777777,
                    'informedness' => 0.1666666702777777,
                    'mcc' => 0.16666666805555555,
                    'cardinality' => 3,
                    'density' => 0.6,
                    'true_positives' => 2,
                    'true_negatives' => 1,
                    'false_positives' => 1,
                    'false_negatives' => 1,
                ],
                'lamb' => [
                    'accuracy' => 0.6,
                    'precision' => 0.5000000025,
                    'recall' => 0.5000000025,
                    'specificity' => 0.6666666677777777,
                    'miss_rate' => 0.4999999975,
                    'fall_out' => 0.3333333322222223,
                    'f1_score' => 0.5000000174999998,
                    'informedness' => 0.1666666702777777,
                    'mcc' => 0.16666666805555555,
                    'cardinality' => 2,
                    'density' => 0.4,
                    'true_positives' => 1,
                    'true_negatives' => 2,
                    'false_positives' => 1,
                    'false_negatives' => 1,
                ],
            ],
        ];
    }

    public function test_build_report()
    {
        $this->assertInstanceOf(MulticlassBreakdown::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $result = $this->report->generate($this->estimator, $this->testing);

        $this->assertEquals($this->outcome, $result);
    }
}
