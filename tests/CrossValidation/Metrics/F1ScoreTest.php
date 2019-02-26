<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\F1Score;
use PHPUnit\Framework\TestCase;

class F1ScoreTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new F1Score();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(F1Score::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertEquals([0, 1], $this->metric->range());

        $this->assertContains(Estimator::CLASSIFIER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::REGRESSOR, $this->metric->compatibility());
        $this->assertNotContains(Estimator::CLUSTERER, $this->metric->compatibility());
        $this->assertContains(Estimator::ANOMALY_DETECTOR, $this->metric->compatibility());
        $this->assertNotContains(Estimator::EMBEDDER, $this->metric->compatibility());
    }

    public function test_score_predictions()
    {
        $predictions = ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'];
        
        $labels = ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'];

        [$min, $max] = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.5833333333423611, $score);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}
