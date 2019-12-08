<?php

namespace Rubix\ML\Tests\CrossValidation\Reports;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Reports\Report;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use PHPUnit\Framework\TestCase;

class AggregateReportTest extends TestCase
{
    /**
     * @var \Rubix\ML\CrossValidation\Reports\AggregateReport
     */
    protected $report;

    public function setUp() : void
    {
        $this->report = new AggregateReport([
            new ConfusionMatrix(),
            new MulticlassBreakdown(),
        ]);
    }

    public function test_build_report() : void
    {
        $this->assertInstanceOf(AggregateReport::class, $this->report);
        $this->assertInstanceOf(Report::class, $this->report);
    }

    public function test_compatibility() : void
    {
        $expected = [
            Estimator::CLASSIFIER,
            Estimator::ANOMALY_DETECTOR,
        ];

        $this->assertEquals($expected, $this->report->compatibility());
    }

    public function test_generate_report() : void
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $result = $this->report->generate($predictions, $labels);

        $this->assertInternalType('array', $result);
        $this->assertCount(2, $result);
    }
}
