<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use PHPUnit\Framework\TestCase;

class VMeasureTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new VMeasure();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(VMeasure::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertEquals([0, 1], $this->metric->range());

        $this->assertNotContains(Estimator::CLASSIFIER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::REGRESSOR, $this->metric->compatibility());
        $this->assertContains(Estimator::CLUSTERER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::ANOMALY_DETECTOR, $this->metric->compatibility());
        $this->assertNotContains(Estimator::EMBEDDER, $this->metric->compatibility());
    }

    public function test_score_predictions()
    {
        $predictions = [1, 2, 2, 1, 2,];

        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        [$min, $max] = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.5833333333513888, $score);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}
