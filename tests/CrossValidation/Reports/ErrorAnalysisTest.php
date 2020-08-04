<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Reports\ErrorAnalysis;
use Rubix\ML\Report;
use Rubix\ML\CrossValidation\Reports\ReportGenerator;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Reports
 * @covers \Rubix\ML\CrossValidation\Reports\ErrorAnalysis
 */
class ErrorAnalysisTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Reports\ErrorAnalysis
     */
    protected $report;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->report = new ErrorAnalysis();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ErrorAnalysis::class, $this->report);
        $this->assertInstanceOf(ReportGenerator::class, $this->report);
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            EstimatorType::regressor(),
        ];

        $this->assertEquals($expected, $this->report->compatibility());
    }

    /**
     * @test
     * @dataProvider generateProvider
     *
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @param (int|float)[] $expected
     */
    public function generate(array $predictions, array $labels, array $expected) : void
    {
        $results = $this->report->generate($predictions, $labels);

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    /**
     * @return \Generator<array>
     */
    public function generateProvider() : Generator
    {
        yield [
            [10, 12, 15, 42, 56, 12, 17, 9, 1, 7],
            [11, 12, 14, 40, 55, 12, 16, 10, 2, 7],
            [
                'mean_absolute_error' => 0.8,
                'median_absolute_error' => 1.0,
                'mean_squared_error' => 1.0,
                'mean_absolute_percentage_error' => 14.02077497665733,
                'rms_error' => 1.0,
                'mean_squared_log_error' => 0.019107097505647368,
                'r_squared' => 0.9958930551562692,
                'error_mean' => -0.2,
                'error_midrange' => -0.5,
                'error_median' => 0.0,
                'error_variance' => 0.9599999999999997,
                'error_mad' => 1.0,
                'error_iqr' => 2.0,
                'error_skewness' => -0.22963966338592326,
                'error_kurtosis' => -1.0520833333333324,
                'error_min' => -2,
                'error_max' => 1,
                'cardinality' => 10,
            ],
        ];

        yield [
            [0.0012, -1.999, -1., 100.2, M_PI],
            [0.0019, -1.822, -0.9, 99.99, M_E],
            [
                'mean_absolute_error' => 0.18220216502615122,
                'median_absolute_error' => 0.17700000000000005,
                'mean_squared_error' => 0.05292430893457563,
                'mean_absolute_percentage_error' => 18.174348688407402,
                'rms_error' => 0.23005283944036775,
                'mean_squared_log_error' => 51.96853354084834,
                'r_squared' => 0.9999669635675313,
                'error_mean' => -0.07112216502615118,
                'error_midrange' => -0.12315541256537399,
                'error_median' => 0.0007000000000000001,
                'error_variance' => 0.04786594657656853,
                'error_mad' => 0.17630000000000004,
                'error_iqr' => 0.455155412565378,
                'error_skewness' => -0.49093461098755187,
                'error_kurtosis' => -1.216490935575394,
                'error_min' => -0.423310825130748,
                'error_max' => 0.17700000000000005,
                'cardinality' => 5,
            ],
        ];
    }
}
