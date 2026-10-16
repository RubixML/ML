<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tuple;
use Rubix\ML\CrossValidation\Metrics\Trustworthiness;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Metrics')]
#[CoversClass(Trustworthiness::class)]
class TrustworthinessTest extends TestCase
{
    protected Trustworthiness $metric;

    protected function setUp() : void
    {
        $this->metric = new Trustworthiness();
    }

    #[Test]
    public function range() : void
    {
        $tuple = $this->metric->range();

        $this->assertInstanceOf(Tuple::class, $tuple);
        $this->assertCount(2, $tuple);
        $this->assertGreaterThan($tuple[0], $tuple[1]);
        $this->assertEquals([0.0, 1.0], $tuple->list());
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertSame([], $this->metric->compatibility());
    }

    #[Test]
    public function zeroNeighborsThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);
        new Trustworthiness(0);
    }

    #[Test]
    public function identitySampleMatchesIdentityEmbedding() : void
    {
        $samples = [[0.0], [1.0], [2.0], [5.0], [10.0], [11.0], [12.0], [20.0]];
        $metric = (new Trustworthiness(2));

        $score = $metric->score($samples, $samples);

        $this->assertEqualsWithDelta(1.0, $score, 1e-8);
    }

    #[Test]
    public function rigidRotationInvariance() : void
    {
        // 8 points on a unit circle, rotated rigidly.
        $n = 8;
        $samples = [];

        for ($i = 0; $i < $n; ++$i) {
            $theta = 2 * M_PI * $i / $n;
            $samples[] = [cos($theta), sin($theta)];
        }

        $angle = 0.7;
        $c = cos($angle);
        $s = sin($angle);
        $embedded = [];

        foreach ($samples as [$x, $y]) {
            $embedded[] = [$x * $c - $y * $s, $x * $s + $y * $c];
        }

        $score = (new Trustworthiness(2))->score($samples, $embedded);

        $this->assertEqualsWithDelta(1.0, $score, 1e-8);
    }

    #[Test]
    public function dimensionReductionLowersScore() : void
    {
        // Points on a 2D circle, projected to 1D (loses information).
        $n = 8;
        $samples = [];
        $embedded = [];

        for ($i = 0; $i < $n; ++$i) {
            $theta = 2 * M_PI * $i / $n;
            $x = cos($theta);
            $y = sin($theta);
            $samples[] = [$x, $y];
            $embedded[] = [$x];
        }

        $good = (new Trustworthiness(2))->score($samples, $samples);
        $bad = (new Trustworthiness(2))->score($samples, $embedded);

        $this->assertGreaterThan($bad, $good);
    }

    #[Test]
    public function mismatchedCountsThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->metric->score([[0.0], [1.0], [2.0]], [[0.0], [1.0]]);
    }

    #[Test]
    public function tooManyNeighborsThrows() : void
    {
        $samples = [[0.0], [1.0], [2.0], [3.0]];
        $this->expectException(InvalidArgumentException::class);
        (new Trustworthiness(2))->score($samples, $samples);
    }

    #[Test]
    public function emptySet() : void
    {
        $this->assertEqualsWithDelta(1.0, $this->metric->score([], []), 1e-8);
    }

    #[Test]
    public function singleSample() : void
    {
        $this->assertEqualsWithDelta(1.0, $this->metric->score([[0.0]], [[0.0]]), 1e-8);
    }

    #[Test]
    public function rangeBoundsEnforced() : void
    {
        $samples = [[0.0], [1.0], [3.0], [4.0], [10.0], [11.0]];
        $embedded = [[0.0], [1.0], [10.0], [4.0], [3.0], [11.0]];

        $score = (new Trustworthiness(1))->score($samples, $embedded);

        [$min, $max] = $this->metric->range()->list();
        $this->assertGreaterThanOrEqual($min, $score);
        $this->assertLessThanOrEqual($max, $score);
    }
}
