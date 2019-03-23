<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\RandIndex;
use PHPUnit\Framework\TestCase;

class RandIndexTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new RandIndex();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(RandIndex::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertEquals([-1, 1], $this->metric->range());

        $this->assertNotContains(Estimator::CLASSIFIER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::REGRESSOR, $this->metric->compatibility());
        $this->assertContains(Estimator::CLUSTERER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::ANOMALY_DETECTOR, $this->metric->compatibility());
        $this->assertNotContains(Estimator::EMBEDDER, $this->metric->compatibility());
    }

    public function test_score_predictions()
    {
        $predictions = [1, 2, 2, 1, 2,];

        $labels = ['lamb', 'lamb', 'wolf', 'lamb', 'wolf'];

        [$min, $max] = $this->metric->range();

        $score = $this->metric->score($predictions, $labels);

        $this->assertEquals(0.16666666666666663, $score);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}
