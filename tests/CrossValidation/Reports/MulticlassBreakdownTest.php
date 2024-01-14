<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\EstimatorType;
use Rubix\ML\Report;
use Rubix\ML\CrossValidation\Reports\ReportGenerator;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Reports
 * @covers \Rubix\ML\CrossValidation\Reports\MulticlassBreakdown
 */
class MulticlassBreakdownTest extends TestCase
{
    /**
     * @var MulticlassBreakdown
     */
    protected $report;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->report = new MulticlassBreakdown();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MulticlassBreakdown::class, $this->report);
        $this->assertInstanceOf(ReportGenerator::class, $this->report);
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];

        $this->assertEquals($expected, $this->report->compatibility());
    }

    /**
     * @test
     * @dataProvider generateProvider
     *
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param mixed[] $expected
     */
    public function generate(array $predictions, array $labels, array $expected) : void
    {
        $results = $this->report->generate($predictions, $labels);

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function generateProvider() : Generator
    {
        yield [
            ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            [
                'overall' => [
                    'accuracy' => 0.6,
                    'balanced accuracy' => 0.5833333333333333,
                    'f1 score' => 0.5833333333333333,
                    'precision' => 0.5833333333333333,
                    'recall' => 0.5833333333333333,
                    'specificity' => 0.5833333333333333,
                    'negative predictive value' => 0.5833333333333333,
                    'false discovery rate' => 0.4166666666666667,
                    'miss rate' => 0.4166666666666667,
                    'fall out' => 0.4166666666666667,
                    'false omission rate' => 0.4166666666666667,
                    'mcc' => 0.16666666666666666,
                    'informedness' => 0.16666666666666652,
                    'markedness' => 0.16666666666666652,
                    'true positives' => 3,
                    'true negatives' => 3,
                    'false positives' => 2,
                    'false negatives' => 2,
                    'cardinality' => 5,
                ],
                'classes' => [
                    'wolf' => [
                        'accuracy' => 0.6,
                        'balanced accuracy' => 0.5833333333333333,
                        'f1 score' => 0.6666666666666666,
                        'precision' => 0.6666666666666666,
                        'recall' => 0.6666666666666666,
                        'specificity' => 0.5,
                        'negative predictive value' => 0.5,
                        'false discovery rate' => 0.33333333333333337,
                        'miss rate' => 0.33333333333333337,
                        'fall out' => 0.5,
                        'false omission rate' => 0.5,
                        'mcc' => 0.16666666666666666,
                        'informedness' => 0.16666666666666652,
                        'markedness' => 0.16666666666666652,
                        'true positives' => 2,
                        'true negatives' => 1,
                        'false positives' => 1,
                        'false negatives' => 1,
                        'cardinality' => 3,
                        'proportion' => 0.6,
                    ],
                    'lamb' => [
                        'accuracy' => 0.6,
                        'balanced accuracy' => 0.5833333333333333,
                        'f1 score' => 0.5,
                        'precision' => 0.5,
                        'recall' => 0.5,
                        'specificity' => 0.6666666666666666,
                        'negative predictive value' => 0.6666666666666666,
                        'false discovery rate' => 0.5,
                        'miss rate' => 0.5,
                        'fall out' => 0.33333333333333337,
                        'false omission rate' => 0.33333333333333337,
                        'mcc' => 0.16666666666666666,
                        'informedness' => 0.16666666666666652,
                        'markedness' => 0.16666666666666652,
                        'true positives' => 1,
                        'true negatives' => 2,
                        'false positives' => 1,
                        'false negatives' => 1,
                        'cardinality' => 2,
                        'proportion' => 0.4,
                    ],
                ],
            ],
        ];

        yield [
            ['tammy', 'tammy', 'tammy', 'tammy'],
            ['tammy', 'tammy', 'morgan', 'tammy'],
            [
                'overall' => [
                    'accuracy' => 0.75,
                    'balanced accuracy' => 0.5,
                    'f1 score' => 0.42857142857142855,
                    'precision' => 0.375,
                    'recall' => 0.5,
                    'specificity' => 0.5,
                    'negative predictive value' => 0.375,
                    'false discovery rate' => 0.625,
                    'miss rate' => 0.5,
                    'fall out' => 0.5,
                    'false omission rate' => 0.625,
                    'mcc' => 0.0,
                    'informedness' => 0.0,
                    'markedness' => -0.25,
                    'true positives' => 3,
                    'true negatives' => 3,
                    'false positives' => 1,
                    'false negatives' => 1,
                    'cardinality' => 4,
                ],
                'classes' => [
                    'tammy' => [
                        'accuracy' => 0.75,
                        'balanced accuracy' => 0.5,
                        'f1 score' => 0.8571428571428571,
                        'precision' => 0.75,
                        'recall' => 1.0,
                        'specificity' => 0.0,
                        'negative predictive value' => 0.0,
                        'false discovery rate' => 0.25,
                        'miss rate' => 0.0,
                        'fall out' => 1.0,
                        'false omission rate' => 1.0,
                        'mcc' => 0.0,
                        'informedness' => 0.0,
                        'markedness' => -0.25,
                        'true positives' => 3,
                        'true negatives' => 0,
                        'false positives' => 1,
                        'false negatives' => 0,
                        'cardinality' => 3,
                        'proportion' => 0.75,
                    ],
                    'morgan' => [
                        'accuracy' => 0.75,
                        'balanced accuracy' => 0.5,
                        'f1 score' => 0.0,
                        'precision' => 0.0,
                        'recall' => 0.0,
                        'specificity' => 1.0,
                        'negative predictive value' => 0.75,
                        'false discovery rate' => 1.0,
                        'miss rate' => 1.0,
                        'fall out' => 0.0,
                        'false omission rate' => 0.25,
                        'mcc' => 0.0,
                        'informedness' => 0.0,
                        'markedness' => -0.25,
                        'true positives' => 0,
                        'true negatives' => 3,
                        'false positives' => 0,
                        'false negatives' => 1,
                        'cardinality' => 1,
                        'proportion' => 0.25,
                    ],
                ],
            ],
        ];
    }
}
