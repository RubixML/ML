<?php

use Rubix\Engine\CrossValidation\Reports\Report;
use Rubix\Engine\CrossValidation\Reports\Classification;
use Rubix\Engine\CrossValidation\Reports\ClassificationReport;
use PHPUnit\Framework\TestCase;

class ClassificationReportTest extends TestCase
{
    protected $report;

    public function setUp()
    {
        $this->report = new ClassificationReport();
    }

    public function test_build_classification_report()
    {
        $this->assertInstanceOf(ClassificationReport::class, $this->report);
        $this->assertInstanceOf(Classification::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_generate_report()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $result = [
            'overall' => [
                'average' => [
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
                'total' => [
                    'cardinality' => 5,
                ],
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

        $report = $this->report->generate($predictions, $labels);

        $this->assertEquals($result, $report);
    }
}
