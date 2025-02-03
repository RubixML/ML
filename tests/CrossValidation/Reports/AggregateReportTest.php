<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Reports;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\EstimatorType;
use Rubix\ML\Report;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use PHPUnit\Framework\TestCase;

#[Group('Reports')]
#[CoversClass(AggregateReport::class)]
class AggregateReportTest extends TestCase
{
    protected AggregateReport $report;

    protected function setUp() : void
    {
        $this->report = new AggregateReport([
            'matrix' => new ConfusionMatrix(),
            'breakdown' => new MulticlassBreakdown(),
        ]);
    }

    public function testCompatibility() : void
    {
        $expected = [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];

        $this->assertEquals($expected, $this->report->compatibility());
    }

    public function testGenerate() : void
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        $result = $this->report->generate(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertInstanceOf(Report::class, $result);
        $this->assertCount(2, $result->toArray());
    }
}
