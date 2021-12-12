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
     * @return \Generator<mixed[]>
     */
    public function generateProvider() : Generator
    {
        yield [
            [10, 12, 15, 42, 56, 12, 17, 9, 1, 7],
            [11, 12, 14, 40, 55, 12, 16, 10, 2, 7],
            [
                'mean absolute error' => 0.8,
                'median absolute error' => 1.0,
                'mean squared error' => 1.0,
                'mean absolute percentage error' => 14.02077497665733,
                'rms error' => 1.0,
                'mean squared log error' => 0.019107097505647368,
                'r squared' => 0.9958930551562692,
                'error mean' => -0.2,
                'error standard deviation' => 0.9797958971132711,
                'error skewness' => -0.22963966338592326,
                'error kurtosis' => -1.0520833333333324,
                'error min' => -2.0,
                'error 25%' => -1.0,
                'error median' => 0.0,
                'error 75%' => 0.75,
                'error max' => 1.0,
                'cardinality' => 10,
            ],
        ];

        yield [
            [0.0012, -1.999, -1., 100.2, M_PI],
            [0.0019, -1.822, -0.9, 99.99, M_E],
            [
                'mean absolute error' => 0.18220216502615122,
                'median absolute error' => 0.17700000000000005,
                'mean squared error' => 0.05292430893457563,
                'mean absolute percentage error' => 18.174348688407402,
                'rms error' => 0.23005283944036775,
                'mean squared log error' => 51.96853354084834,
                'r squared' => 0.9999669635675313,
                'error mean' => -0.07112216502615118,
                'error standard deviation' => 0.2187828754189151,
                'error skewness' => -0.49093461098755187,
                'error kurtosis' => -1.216490935575394,
                'error min' => -0.423310825130748,
                'error 25%' => -0.21000000000000796,
                'error median' => 0.0007000000000000001,
                'error 75%' => 0.09999999999999998,
                'error max' => 0.17700000000000005,
                'cardinality' => 5,
            ],
        ];
    }
}
