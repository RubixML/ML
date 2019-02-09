<?php

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Homogeneity;
use PHPUnit\Framework\TestCase;

class HomogeneityTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->metric = new Homogeneity();
    }

    public function test_build_metric()
    {
        $this->assertInstanceOf(Homogeneity::class, $this->metric);
        $this->assertInstanceOf(Metric::class, $this->metric);

        $this->assertEquals([0, 1], $this->metric->range());

        $this->assertNotContains(Estimator::CLASSIFIER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::REGRESSOR, $this->metric->compatibility());
        $this->assertContains(Estimator::CLUSTERER, $this->metric->compatibility());
        $this->assertNotContains(Estimator::DETECTOR, $this->metric->compatibility());
        $this->assertNotContains(Estimator::EMBEDDER, $this->metric->compatibility());
    }

    public function test_score_predictions()
    {
        $predictions = [1, 2, 2, 1, 2];
        
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
